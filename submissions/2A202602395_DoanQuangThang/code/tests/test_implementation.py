"""CPU checks for arithmetic, leakage, frozen BN, timing and real training/checkpoint I/O."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch import nn

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parents[2]
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(ROOT))
from deepweeds_lab import losses, model, inference, dataset, benchmark, train
from eval import compute_metrics, read_pred, check_against_csv


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(4, 9)
        self.pretrained_cfg = {'mean': dataset.IMAGENET_MEAN, 'std': dataset.IMAGENET_STD, 'tag': 'synthetic-test'}
    def get_classifier(self):
        return self.head
    def forward(self, x):
        return self.head(self.body(x).flatten(1))


class TestImplementation(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); train.set_seed(4)

    def test_focal_zero_and_label_smoothing_zero_equal_ce(self):
        z, y = torch.randn(11, 9), torch.arange(11) % 9
        expected = nn.CrossEntropyLoss()(z, y)
        self.assertTrue(torch.allclose(losses.FocalLoss(0)(z, y), expected, atol=1e-6))
        self.assertTrue(torch.allclose(losses.LabelSmoothingCE(0)(z, y), expected, atol=1e-6))
        self.assertTrue(torch.isfinite(losses.FocalLoss(2)(z, y)))

    def test_cutmix_area_and_labels_match_pixels(self):
        x = torch.stack([torch.full((3, 17, 19), float(i)) for i in range(12)])
        y = torch.arange(12)
        for _ in range(8):
            mixed, (a, b, lam) = losses.mix_batch(x, y, mode='cutmix')
            self.assertTrue(torch.equal(a, y)); self.assertGreaterEqual(lam, 0); self.assertLessEqual(lam, 1)
            changed = b != a
            if changed.any():
                fraction = (mixed[changed, 0] != x[changed, 0]).float().mean().item()
                self.assertAlmostEqual(fraction, 1 - lam, places=6)
            self.assertTrue(torch.equal(x, torch.stack([torch.full((3, 17, 19), float(i)) for i in range(12)])))

    def test_mixup_uses_same_permutation_for_labels(self):
        x = torch.arange(12.).reshape(12, 1, 1, 1); y = torch.arange(12)
        mixed, (a, b, lam) = losses.mix_batch(x, y, mode='mixup')
        self.assertTrue(torch.allclose(mixed.flatten(), lam * a + (1 - lam) * b))
        z = torch.randn(12, 9); a, b = a % 9, b % 9
        actual = losses.mixed_loss(nn.CrossEntropyLoss(), z, (a, b, lam))
        expected = lam * nn.CrossEntropyLoss()(z, a) + (1-lam) * nn.CrossEntropyLoss()(z, b)
        self.assertTrue(torch.allclose(actual, expected))

    def test_optimizer_groups_cover_once_and_exclude_all_bias_norm(self):
        m = Tiny(); groups = model.param_groups(m, 1e-4, 1e-3, .05)
        params = [p for g in groups for p in g['params']]
        self.assertEqual(len(params), len(set(map(id, params))))
        self.assertEqual(set(map(id, params)), set(map(id, m.parameters())))
        for g in groups:
            for p in g['params']:
                if p.ndim <= 1:
                    self.assertEqual(g['weight_decay'], 0)
                if id(p) in {id(q) for q in m.head.parameters()}:
                    self.assertEqual(g['lr'], 1e-3)

    def test_frozen_bn_and_parameters_do_not_change(self):
        m = Tiny(); model.freeze_backbone(m); before = copy.deepcopy(m.body.state_dict())
        cfg = train.Config(init='frozen', amp=False, device='cpu', epochs=1)
        loader = [(torch.randn(9, 3, 8, 8), torch.arange(9), ['a'] * 9)]
        opt = train.build_optimizer(m, cfg); sch = train.build_scheduler(opt, cfg, 1)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        train.train_one_epoch(m, loader, nn.CrossEntropyLoss(), opt, sch, scaler, cfg, 'cpu')
        for name, p in m.body.state_dict().items():
            self.assertTrue(torch.equal(before[name], p), name)

    def test_fusion_preserves_outputs_and_original(self):
        m = Tiny().eval(); x = torch.randn(5, 3, 16, 16)
        fused = inference.fuse_conv_bn(m)
        self.assertTrue(torch.allclose(m(x), fused(x), atol=1e-5))
        self.assertIsInstance(m.body[1], nn.BatchNorm2d)
        self.assertIsInstance(fused.body[1], nn.Identity)

    def test_ema_copies_bn_buffers(self):
        m = Tiny(); ema = train.EMA(m, .9)
        m.train(); m(torch.randn(5, 3, 8, 8)); ema.update(m)
        self.assertTrue(torch.equal(m.body[1].running_mean, ema.model.body[1].running_mean))

    def test_calibration_preserves_argmax_and_does_not_increase_val_nll(self):
        rng = np.random.default_rng(2); z = rng.normal(size=(180, 9)) * 5; y = rng.integers(0, 9, 180)
        T = inference.fit_temperature(z, y)
        before, after = inference.apply_temperature(z, 1), inference.apply_temperature(z, T)
        self.assertTrue(np.array_equal(before.argmax(1), after.argmax(1)))
        self.assertLessEqual(compute_metrics(y, after.argmax(1), after)['nll'], compute_metrics(y, before.argmax(1), before)['nll'])
        np.testing.assert_allclose(after.sum(1), 1)

    def test_views_and_aggregation(self):
        x = torch.arange(3 * 8 * 8).reshape(1, 3, 8, 8).float()
        self.assertTrue(torch.equal(inference.view_hflip(inference.view_hflip(x)), x))
        self.assertEqual(len(inference.views_multicrop(x, 6)), 5)
        zs = [np.zeros((5, 9)), np.arange(45).reshape(5, 9)]
        for space in ('prob', 'logit'):
            np.testing.assert_allclose(inference.aggregate_views(zs, space).sum(1), 1)

    def test_timing_sync_and_minimum_samples(self):
        calls = {'fn': 0, 'sync': 0}
        def fn(): calls['fn'] += 1
        def sync(): calls['sync'] += 1
        report = benchmark.bench(fn, 10, 50, sync)
        self.assertEqual(calls, {'fn': 60, 'sync': 101}); self.assertEqual(report['n'], 50)
        self.assertLessEqual(report['p50'], report['p95']); self.assertLessEqual(report['p95'], report['p99'])
        with self.assertRaises(ValueError): benchmark.bench(fn, 0, 1)

    def test_split_rejects_overlap_and_bad_count(self):
        a = pd.DataFrame({'Filename': ['a.jpg'], 'Label': [0]})
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            dataset.check_split(a, a, a, '.')
        with self.assertRaisesRegex(ValueError, '17509'):
            dataset.check_split(a, a.assign(Filename='b.jpg'), a.assign(Filename='c.jpg'), '.')

    def test_cli_types_and_defaults(self):
        c = train.Config(**train.parse_overrides(['seed=2', 'ema_decay=0.99', 'amp=false', 'mix=none']))
        self.assertEqual(c.seed, 2); self.assertEqual(c.ema_decay, .99); self.assertFalse(c.amp); self.assertIsNone(c.mix)
        self.assertFalse(train.Config().save_test_predictions)
        with self.assertRaises(ValueError): train.parse_overrides(['amp=anything'])
        with self.assertRaises(ValueError): train.parse_overrides(['typo=2'])

    def test_real_train_artifacts_resume_and_test_gate(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); images = root / 'images'; images.mkdir()
            dfs = []
            for split in ('train', 'val', 'test'):
                records = []
                for y in range(9):
                    name = f'{split}{y}.jpg'; Image.new('RGB', (20, 20), (y*25, 80, 100)).save(images / name)
                    records.append({'Filename': name, 'Label': y})
                dfs.append(pd.DataFrame(records))
            cfg = train.Config(exp_id='SYNTHETIC', epochs=2, batch_size=9, img_size=16, images_dir=str(images),
                labels_dir=str(root), out_dir=str(root / 'runs'), pred_dir=str(root / 'predictions'),
                curves_dir=str(root / 'curves'), num_workers=0, amp=False, device='cpu')
            with patch.object(dataset, 'load_split', return_value=tuple(dfs)), patch.object(dataset, 'check_split', return_value={'fixture': True}), patch.object(model, 'build_model', side_effect=lambda *a, **kw: Tiny()), patch.object(model, 'count_gmacs', return_value=0.0):
                result = train.run(cfg)
                self.assertFalse((root / 'predictions/SYNTHETIC_seed0_test.csv').exists())
                self.assertTrue((root / 'curves/SYNTHETIC_seed0.png').exists())
                self.assertEqual(len(pd.read_csv(train.run_dir(cfg) / 'history.csv')), 2)
                pd.DataFrame(dfs[1]).to_csv(root / 'val.csv', index=False)
                check_against_csv(read_pred(str(train.pred_path(cfg, 'val'))), str(root / 'val.csv'), 'val')
                self.assertEqual(train.run(cfg), result)
                with self.assertRaisesRegex(ValueError, 'different experiment'):
                    train.run(replace(cfg, loss='focal'))
                test_cfg = replace(cfg, exp_id='SYNTHETICFINAL', save_test_predictions=True)
                result = train.run(test_cfg)
                self.assertEqual(result['n_test'], 9)
                self.assertTrue(train.pred_path(test_cfg, 'test').exists())
                self.assertEqual(train.run(test_cfg), result)

    def test_final_lock_calibration_cache_and_reporting_artifacts(self):
        from deepweeds_lab.experiments import final_predictions
        from deepweeds_lab.reporting import create_artifacts
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'curves').mkdir(); (root / 'images').mkdir()
            cfg = train.Config(exp_id='T09', device='cpu', num_workers=0, img_size=16, batch_size=9,
                               curves_dir=str(root / 'curves'), images_dir=str(root / 'images'))
            frames = []
            for split in ('train', 'val', 'test'):
                records = []
                for y in range(9):
                    name = f'{split}{y}.jpg'; Image.new('RGB', (20,20), (y*25,50,80)).save(root / 'images' / name)
                    records.append({'Filename': name, 'Label': y})
                frames.append(pd.DataFrame(records))
            loaded_test = []
            def loader(c, df, method):
                if df.Filename.iloc[0].startswith('test'):
                    locked = root / 'final_cache' / current[0] / f'seed{c.seed}' / 'locked_seed_config.json'
                    self.assertTrue(locked.exists(), 'Must lock temperature before test loader')
                    loaded_test.append((current[0], c.seed))
                return [(torch.randn(9,3,16,16), torch.arange(9), df.Filename.tolist())]
            final_rows = []; current = ['F01']
            with patch.object(dataset, 'load_split', return_value=tuple(frames)), patch('deepweeds_lab.experiments.load_checkpoint_model', side_effect=lambda c: Tiny().eval()), patch('deepweeds_lab.experiments.strategy_loader', side_effect=loader):
                for tag in ('F01','T00','R01'):
                    current[0] = tag
                    for seed in (0,1,2):
                        c = replace(cfg, seed=seed)
                        Image.new('RGB',(20,20)).save(root / 'curves' / f'T09_seed{seed}.png')
                        r = final_predictions(c, 'single', tag, root)
                        final_rows.append(r)
                        self.assertEqual(final_predictions(c, 'single', tag, root), r)
            self.assertEqual(len(loaded_test), 9, 'One test forward per configuration and seed, no repeat')
            for seed in (0,1,2):
                pred = read_pred(str(root / 'predictions' / f'F01_seed{seed}_test.csv'))
                uncal = read_pred(str(root / 'predictions' / f'F01uncal_seed{seed}_test.csv'))
                np.testing.assert_array_equal(pred.y_pred, uncal.y_pred)
            # Entire reporting path exercised on labeled SYNTHETIC fixtures only.
            rows = [{'exp_id':'B01','backbone':'SYNTHETIC','pretrained_tag':'synthetic','params_m':.1,'gmac':.01,
                     'macro_f1_val':.2,'top1_val':.3,'train_seconds_per_epoch':1.,'latency_ms':1.}]
            training = [dict(rows[0],exp_id='T00',axis='baseline',changes='{}'), dict(rows[0],exp_id='T09',axis='combined',changes='{}')]
            inference_rows = []
            for index, method in enumerate(('single','temperature')):
                timing = dict(gpu='SYNTHETIC CPU',dtype='fp32',batch=32,img_size=16,p50=1.,p95=2.,p99=3.,images_per_s=32.,torch=torch.__version__,preprocessing=False,fused_bn=False)
                inference_rows.append(dict(exp_id=f'I0{index}',method=method,K=1,macro_f1=.2,top1=.3,ece=.1,p50=1.,p95=2.,p99=3.,relative_cost=1.,gpu='SYNTHETIC CPU',torch=torch.__version__,dtype='fp32',batch=1,img_size=16,images_per_s=1.,preprocessing=False,fused_bn=False,latency_batch32=timing))
            selection = dict(backbone='SYNTHETIC',training_config=train.asdict(cfg),inference='single',seeds=[0,1,2],realtime_method='single')
            create_artifacts(root,rows,training,inference_rows,final_rows,selection)
            with pd.ExcelFile(root / 'results.xlsx') as workbook:
                self.assertEqual(workbook.sheet_names, ['Backbones','Training','Inference','Final','PerClass','Latency','Summary'])
            self.assertTrue((root / 'report.md').exists()); self.assertTrue((root / 'confusion_matrix.png').exists())
            final_sheet=pd.read_excel(root / 'results.xlsx',sheet_name='Final')
            summary=final_sheet[final_sheet.exp_id=='F01'].iloc[-1]
            mean,std=np.mean([r['macro_f1_test'] for r in final_rows if r['exp_id']=='F01']),np.std([r['macro_f1_test'] for r in final_rows if r['exp_id']=='F01'],ddof=1)
            self.assertAlmostEqual(summary.macro_f1_test, mean); self.assertAlmostEqual(summary.macro_f1_test_std, std)

    def test_upstream_label_discrepancy_is_reported_without_relabeling(self):
        from deepweeds_lab.prepare import compare_catalog
        frames = [pd.DataFrame({'Filename':['a.jpg'],'Label':[0]}),
                  pd.DataFrame({'Filename':['b.jpg'],'Label':[7]}),
                  pd.DataFrame({'Filename':['c.jpg'],'Label':[8]})]
        original = pd.DataFrame({'Filename':['a.jpg','b.jpg','c.jpg'],'Label':[1,7,8]})
        before = [d.copy(deep=True) for d in frames]
        with tempfile.TemporaryDirectory() as folder:
            differences = compare_catalog(frames, original, folder)
            self.assertEqual(differences, [{'Filename':'a.jpg','Label_split':0,'Label_catalog':1}])
            self.assertEqual(json.loads((Path(folder)/'catalog_label_discrepancies.json').read_text())['count'],1)
            for a,b in zip(frames,before): pd.testing.assert_frame_equal(a,b)
            with self.assertRaisesRegex(ValueError, 'filenames'):
                compare_catalog(frames, original.iloc[:-1], folder)

    def test_inference_trials_merge_real_benchmark_schema_and_resume(self):
        from deepweeds_lab.experiments import inference_trials
        frames = tuple(pd.DataFrame({'Filename':[f'{split}{i}.jpg' for i in range(9)], 'Label':range(9)}) for split in ('train','val','test'))
        def loader(cfg, df, method):
            return [(torch.randn(9,3,32,32), torch.arange(9), df.Filename.tolist())]
        def timing(model, batch_size, img_size, method='single', **kw):
            return dict(method=method, gpu='SYNTHETIC CPU', dtype='fp32', batch=batch_size,
                        img_size=img_size, p50=1., p95=2., p99=3., mean=1., n=50, warmup=10,
                        images_per_s=float(batch_size*1000), torch=torch.__version__, preprocessing=False,
                        fused_bn=method=='fused', temperature=kw.get('temperature',1.))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); cfg=train.Config(img_size=16, batch_size=9, device='cpu', num_workers=0)
            with patch.object(dataset,'load_split',return_value=frames), patch('deepweeds_lab.experiments.load_checkpoint_model',side_effect=lambda c:Tiny().eval()), patch('deepweeds_lab.experiments.strategy_loader',side_effect=loader), patch('deepweeds_lab.experiments.strategy_latency',side_effect=timing):
                rows=inference_trials(cfg,root)
                self.assertEqual(len(rows),8)
                self.assertEqual(len({r['method'] for r in rows}),8)
                self.assertTrue(all(r['n']==50 and r['latency_batch32']['batch']==32 for r in rows))
                self.assertEqual(json.loads((root/'inference_results.json').read_text()), rows)
                with patch('deepweeds_lab.experiments.predict_strategy',side_effect=AssertionError('Completed inference must not be repeated')):
                    self.assertEqual(inference_trials(cfg,root),rows)

    def test_private_recovery_downloads_verify_hashes_and_hide_urls(self):
        from deepweeds_lab.recovery import restore_download_manifest
        import io, hashlib, base64, zipfile
        payload=b'SYNTHETIC checkpoint download'
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); manifest=root/'manifest.json'; staging=root/'staging'
            entry={'path':'lab_output/checkpoint.pt','size':len(payload),
                   'sha256':hashlib.sha256(payload).hexdigest(),
                   'url':'https://www.kaggleusercontent.com/private?token=SYNTHETIC_SECRET'}
            record={'source':'thngonquang/deepweeds-day2','version':3,'files':[entry]}
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w') as archive:
                archive.writestr('lab_output/config.json','{"synthetic":true}')
            record.update(metadata_zip_b64=base64.b64encode(buffer.getvalue()).decode(),
                          metadata_zip_sha256=hashlib.sha256(buffer.getvalue()).hexdigest())
            manifest.write_text(json.dumps(record))
            with patch('deepweeds_lab.recovery.urlopen',return_value=io.BytesIO(payload)) as download:
                restore_download_manifest(manifest,staging)
                self.assertEqual((staging/entry['path']).read_bytes(),payload)
                self.assertTrue(json.loads((staging/'lab_output/config.json').read_text())['synthetic'])
                restore_download_manifest(manifest,staging)
                self.assertEqual(download.call_count,1)
            with patch('deepweeds_lab.recovery.urlopen',side_effect=RuntimeError(entry['url'])):
                with self.assertRaisesRegex(RuntimeError,'refresh private recovery links') as error:
                    restore_download_manifest(manifest,root/'failed')
                self.assertNotIn('SYNTHETIC_SECRET',str(error.exception))
            entry['sha256']='0'*64; manifest.write_text(json.dumps(record))
            with patch('deepweeds_lab.recovery.urlopen',return_value=io.BytesIO(payload)):
                with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                    restore_download_manifest(manifest,root/'bad_hash')
            entry['path']='../escape'; manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'Unsafe recovery manifest'):
                restore_download_manifest(manifest,root/'unsafe')

    def test_recovery_requires_complete_evidence_and_never_replaces_data(self):
        from deepweeds_lab.recovery import restore_completed_training, restore_kaggle_input
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); source=base/'input'/'previous'/'lab_output'; source.mkdir(parents=True)
            for name,ids in [('backbone_results.json',[f'B{i:02d}' for i in range(1,6)]),('training_results.json',[f'T{i:02d}' for i in range(10)])]:
                rows=[]
                for exp_id in ids:
                    cfg={'exp_id':exp_id,'seed':0}; rows.append(dict(exp_id=exp_id,seed=0,config=cfg))
                    runfolder=source/'runs'/exp_id/'seed0'; runfolder.mkdir(parents=True)
                    for filename in ('best.pt','val_logits.npz','history.csv','pretrained.json'):
                        (runfolder/filename).write_text('SYNTHETIC recovery fixture')
                    for filename in ('config.json','result.json'):
                        (runfolder/filename).write_text(json.dumps(cfg))
                (source/name).write_text(json.dumps(rows))
            (source/'execution_error.json').write_text(json.dumps({'type':'TypeError','message':'synthetic prior failure'}))
            destination=base/'working'/'lab_output'
            result=restore_kaggle_input(destination,base/'input')
            self.assertEqual((result['completed_backbones'],result['completed_training_configs']),(5,10))
            self.assertTrue((destination/'prior_errors/version3_execution_error.json').exists())
            self.assertFalse((destination/'execution_error.json').exists())
            self.assertEqual(restore_completed_training(source,destination)['copied_files'],0)
            import zipfile
            archive_input=base/'archive_input'; archive_input.mkdir()
            with zipfile.ZipFile(archive_input/'lab_output_v3.zip','w') as archive:
                for path in source.rglob('*'):
                    if path.is_file(): archive.write(path, path.relative_to(source.parent))
            restored=restore_kaggle_input(base/'archive_working'/'lab_output',archive_input)
            self.assertEqual(restored['completed_training_configs'],10)
            unsafe_input=base/'unsafe_input'; unsafe_input.mkdir()
            with zipfile.ZipFile(unsafe_input/'lab_output_v3.zip','w') as archive:
                archive.writestr('../escape.txt','must not extract')
            with self.assertRaisesRegex(ValueError,'Unsafe recovery archive'):
                restore_kaggle_input(base/'unsafe_working'/'lab_output',unsafe_input)
            self.assertFalse((base/'unsafe_working'/'escape.txt').exists())
            protected=destination/'runs/B01/seed0/best.pt'; protected.write_text('USER DATA DO NOT REPLACE')
            with self.assertRaisesRegex(ValueError,'Refusing to overwrite'):
                restore_completed_training(source,destination)
            self.assertEqual(protected.read_text(),'USER DATA DO NOT REPLACE')
            with self.assertRaisesRegex(RuntimeError,'refusing full retraining'):
                restore_kaggle_input(base/'empty',base/'missing')
            with self.assertRaises(ValueError):
                (source/'backbone_results.json').write_text('[]')
                restore_completed_training(source,base/'other')


if __name__ == '__main__':
    unittest.main()
