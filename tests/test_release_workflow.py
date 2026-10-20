"""Public quickstart contract: offline previews cannot silently launch 50B work."""
import json,subprocess,sys


def test_preview_defaults_to_one_gpu_and_hardware_qualification(tmp_path):
    root=tmp_path/'unused'
    r=subprocess.run([sys.executable,'-m','scripts.cloud.reproduce','--data-dir',str(root),'--plan'],capture_output=True,text=True,check=True)
    d=json.loads(r.stdout)
    assert d['gpus']==1
    assert d['config']['target_tokens']==10_000_000_000
    assert d['architecture']=='engram' and d['qualification_state_reused'] is False
    assert any('qualification' in x for x in d['stages'])
    assert not root.exists()


def test_50b_cannot_be_advertised_as_the_old_two_source_workflow(tmp_path):
    root=tmp_path/'unused'
    r=subprocess.run([sys.executable,'-m','scripts.cloud.reproduce','--data-dir',str(root),'--through','50B'],capture_output=True,text=True)
    assert r.returncode!=0 and not root.exists()


def test_5090_preview_preserves_global_batch_and_gpu_count(tmp_path):
    root=tmp_path/'unused'
    r=subprocess.run([sys.executable,'-m','scripts.cloud.reproduce','--data-dir',str(root),
        '--hardware-profile','5090-32','--gpus','4','--plan'],capture_output=True,text=True,check=True)
    d=json.loads(r.stdout)
    assert d['gpus']==4 and d['config']['gradient_accumulation_steps']==16
    assert not root.exists()
