"""Local-only Web UI entry point. It never exposes the service on the LAN."""

from __future__ import annotations

import platform
import threading
import webbrowser
import json
import subprocess
import socket
import os
import signal
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .jobs import JobStore
from .audit import audit_job
from .assemble import assemble_raw_markdown
from .config import LlmSettingsStore
from .ocr import run_ocr_job
from .llm import LlmProfile, ProofreadPaused
from .proofread import proofread_job


class CreateJobRequest(BaseModel):
    pdf_path: str
    output_dir: str = "output/jobs"
    batch_size: int = Field(default=50, ge=1, le=200)


class RunJobRequest(BaseModel):
    max_pages: int | None = Field(default=None, ge=1)
    vl_backend: str | None = None
    vl_server_url: str | None = None
    auto_proofread: bool = False
    cpu_fallback: bool = True


class ProofreadRequest(BaseModel):
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    max_pages: int | None = Field(default=None, ge=1)


class SaveLlmSettingsRequest(BaseModel):
    base_url: str
    model: str
    api_key: str = ""
    auto_proofread: bool = False


_INDEX_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>BookOCR</title>
<style>body{max-width:760px;margin:48px auto;font:16px -apple-system,sans-serif;color:#202124}input{width:100%;box-sizing:border-box;margin:6px 0 16px;padding:10px}input[type=checkbox]{width:auto;margin:0 8px 0 0}button{padding:10px 16px;margin:0 6px 16px 0}button:disabled{opacity:.6}pre{padding:12px;background:#f6f8fa;white-space:pre-wrap}h2{margin-top:36px}.hint{color:#5f6368;font-size:14px}details{margin:8px 0 16px}progress{width:100%;height:18px;margin:4px 0}.action-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.action-status{margin:0 0 16px;color:#5f6368}.action-status.ok{color:#137333}.action-status.error{color:#b3261e}.action-progress{margin:-6px 0 12px}[hidden]{display:none!important}</style></head>
<body><h1>BookOCR</h1><button style="float:right" onclick="shutdown()">停止 BookOCR 服务</button><p>本地运行，PDF 不会上传；模型配置保存于本机，API Key 保存到 macOS 钥匙串。</p>
<h2>创建 / 继续 OCR</h2><label>扫描 PDF</label><input id="pdf" placeholder="点击“浏览 PDF”选择文件；也可粘贴路径"><button onclick="pickPdf()">浏览 PDF…</button><label>输出目录</label><input id="output-dir" value="output/jobs"><button onclick="pickOutputDir()">选择输出目录…</button><button onclick="createJob()">创建任务</button><p class="hint" id="ocr-note">创建成功后会自动填入任务 ID。</p><label>任务 ID</label><input id="job" placeholder="创建后会自动填入这里"><label><input id="use-mlx" type="checkbox">使用本机 Apple 芯片加速</label><button onclick="startMlx()">启动 Apple 加速</button><p class="hint" id="mlx-note">首次使用时先不要勾选加速，先试跑 1 页以准备本地模型；之后点击“启动 Apple 加速”并勾选它即可加快全书识别。</p><details><summary>高级设置</summary><label>本机加速服务地址</label><input id="vl-url" value="http://127.0.0.1:8118/v1"><p class="hint">只有排障或改端口时才需要修改。</p></details><label>本次 OCR 页数（可选，首次建议填 1）</label><input id="ocr-max" type="number" min="1" placeholder="留空=处理全部待处理页"><button id="run-button" onclick="runJob()">开始识别</button><button onclick="statusJob()">查看进度</button><progress id="ocr-progress" value="0" max="1"></progress><p class="hint" id="ocr-progress-note">尚未开始识别。</p>
<h2>导出与质量检查</h2><div class="action-row"><button id="assemble-button" onclick="assemble()">生成原始 Markdown</button><span id="assemble-status" class="action-status" aria-live="polite">尚未生成</span></div><progress id="assemble-progress" class="action-progress" hidden></progress><div class="action-row"><button id="audit-button" onclick="audit()">运行质量检查</button><span id="audit-status" class="action-status" aria-live="polite">尚未检查</span></div><progress id="audit-progress" class="action-progress" hidden></progress>
<h2>LLM 保守校对</h2><p class="hint">默认示例是 DeepSeek；可替换为 Kimi、GPT 或任何兼容 OpenAI Chat Completions 的模型服务。</p><label>Base URL（兼容 /chat/completions）</label><input id="base" placeholder="https://你的服务"><label>模型名</label><input id="model-name" placeholder="你的模型名"><label>API Key（保存后显示为空，已存入钥匙串）</label><input id="key" type="password"><label><input id="auto-proofread" type="checkbox">OCR 全部完成后自动开始校对（会使用已保存模型的额度）</label><button onclick="saveConfig()">保存模型配置</button><p class="hint" id="config-note">首次填写后点击“保存模型配置”；以后会自动读取本机配置和钥匙串 Key。</p><label>先校对前几页（可选，空白=全书）</label><input id="proofread-max" type="number" min="1" placeholder="例如 3"><div class="action-row"><button id="proofread-button" onclick="proofread()">立即开始校对</button><button id="proofread-pause-button" onclick="pauseProofread()" hidden>暂停校对</button><button id="proofread-resume-button" onclick="resumeProofread()" hidden>继续校对</button><button id="proofread-status-button" onclick="proofreadStatus()">刷新校对状态</button><span id="proofread-status" class="action-status" aria-live="polite">尚未开始校对</span></div><progress id="proofread-progress" class="action-progress" value="0" max="1" hidden></progress><details><summary>查看技术详情（仅排障时需要）</summary><pre id="result">尚未创建任务</pre></details>
<script>
const result=document.querySelector('#result');
const val=id=>document.querySelector(id).value;
const optionalNumber=id=>{const n=val(id);return n?Number(n):null};
async function show(response){const body=await response.json();result.textContent=JSON.stringify(body,null,2);return body}
function renderOcrStatus(data){const pages=data.pages||{};const total=data.total_pages||0;const complete=pages.complete||0;const running=pages.running||0;const pending=pages.pending||0;const failed=pages.failed||0;const ocr=data.ocr_status||{};const progress=document.querySelector('#ocr-progress');const note=document.querySelector('#ocr-progress-note');const run=document.querySelector('#run-button');if(!total){progress.max=1;progress.value=0;note.textContent='请先创建任务。';run.textContent='开始识别';run.disabled=false;return}if(ocr.status==='retrying_cpu'){const retryTotal=Number(ocr.retry_total||0);const retryDone=Number(ocr.retry_completed||0);progress.max=retryTotal||1;progress.value=retryDone;note.textContent='兼容重试进度：'+retryDone+'/'+retryTotal+' 页（全书已完成 '+complete+'/'+total+' 页；剩余 '+running+' 页）。'}else{progress.max=total||1;progress.value=complete;if(running){note.textContent='全书识别进度：'+complete+'/'+total+' 页；正在识别 '+running+' 页'+(failed?'；已有 '+failed+' 页将自动兼容重试':'')+'。'}else if(failed){note.textContent='已完成 '+complete+'/'+total+' 页；仍有 '+failed+' 页失败，需要兼容重试。'}else if(pending){note.textContent='已完成 '+complete+'/'+total+' 页；待处理 '+pending+' 页。'}else{note.textContent='OCR 已完成：'+complete+'/'+total+' 页。'}}run.disabled=Boolean(running)||ocr.status==='retrying_cpu';run.textContent=run.disabled?'识别进行中…':complete>=total?'识别已完成':complete?'继续识别剩余页面':'开始识别';}
async function loadConfig(){const c=await (await fetch('/api/config/llm')).json();document.querySelector('#base').value=c.base_url;document.querySelector('#model-name').value=c.model;document.querySelector('#auto-proofread').checked=c.auto_proofread;document.querySelector('#config-note').textContent=c.key_configured?'已加载本机模型配置与钥匙串 Key。':'请填写并保存模型配置。';let savedJob=localStorage.getItem('bookocr.lastJobId');if(!savedJob){const latest=await (await fetch('/api/latest-job')).json();savedJob=latest.id||''}if(savedJob){document.querySelector('#job').value=savedJob;statusJob();proofreadStatus()}}
async function saveConfig(){const r=await fetch('/api/config/llm',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({base_url:val('#base'),model:val('#model-name'),api_key:val('#key'),auto_proofread:document.querySelector('#auto-proofread').checked})});const data=await show(r);if(data.key_configured){document.querySelector('#key').value='';document.querySelector('#config-note').textContent='保存成功：已写入本机配置和 macOS 钥匙串。'} }
async function pickPdf(){const data=await show(await fetch('/api/pick-pdf',{method:'POST'}));if(data.path)document.querySelector('#pdf').value=data.path}
async function pickOutputDir(){const data=await show(await fetch('/api/pick-output-dir',{method:'POST'}));if(data.path)document.querySelector('#output-dir').value=data.path}
async function startMlx(){const data=await show(await fetch('/api/mlx/start',{method:'POST'}));document.querySelector('#mlx-note').textContent=data.message||'正在启动 Apple 加速，请等待终端窗口显示服务已就绪。';waitForMlx()}
async function waitForMlx(){const data=await (await fetch('/api/mlx/status')).json();if(data.running){document.querySelector('#use-mlx').checked=true;document.querySelector('#mlx-note').textContent='Apple 加速已就绪，已自动启用。'}else setTimeout(waitForMlx,1500)}
async function createJob(){const data=await show(await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pdf_path:val('#pdf'),output_dir:val('#output-dir')})}));const note=document.querySelector('#ocr-note');if(data.id){document.querySelector('#job').value=data.id;localStorage.setItem('bookocr.lastJobId',data.id);note.textContent='任务已创建：共 '+data.total_pages+' 页。任务 ID 已自动填入。';renderOcrStatus({total_pages:data.total_pages,pages:{pending:data.total_pages}})}else{note.textContent='创建失败：请查看页面底部的错误信息。'}}
async function runJob(){const useMlx=document.querySelector('#use-mlx').checked;const url=val('#vl-url');const max_pages=optionalNumber('#ocr-max');const body={auto_proofread:document.querySelector('#auto-proofread').checked};if(useMlx){body.vl_backend='mlx-vlm-server';body.vl_server_url=url}if(max_pages)body.max_pages=max_pages;const data=await show(await fetch('/api/jobs/'+val('#job')+'/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}));document.querySelector('#ocr-note').textContent=data.status==='started'?'识别已开始，进度会自动更新。':'无法开始识别，请查看页面底部信息。';setTimeout(statusJob,500)}
async function statusJob(){const data=await show(await fetch('/api/jobs/'+val('#job')));renderOcrStatus(data);const pages=data.pages||{};if(data.active||(pages.running||0)>0)setTimeout(statusJob,2000)}
function startAction(name,message){const button=document.querySelector('#'+name+'-button');const status=document.querySelector('#'+name+'-status');button.disabled=true;status.className='action-status';status.textContent=message;document.querySelector('#'+name+'-progress').hidden=false}
function finishAction(name,message,ok){const button=document.querySelector('#'+name+'-button');const status=document.querySelector('#'+name+'-status');button.disabled=false;status.className='action-status '+(ok?'ok':'error');status.textContent=message;document.querySelector('#'+name+'-progress').hidden=true}
async function assemble(){startAction('assemble','正在生成，请稍候…');try{const response=await fetch('/api/jobs/'+val('#job')+'/assemble',{method:'POST'});const data=await show(response);if(response.ok&&data.markdown){const filename=data.markdown.split('/').pop();finishAction('assemble','✓ 已生成：'+filename+'（保存在所选输出文件夹）',true)}else{finishAction('assemble','生成失败：'+(data.detail||'请查看技术详情'),false)}}catch(error){result.textContent=String(error);finishAction('assemble','无法生成，请确认本地服务仍在运行。',false)}}
async function audit(){startAction('audit','正在检查整本书，请稍候…');try{const response=await fetch('/api/jobs/'+val('#job')+'/audit');const data=await show(response);if(response.ok&&data.data){const report=data.data;const review=(report.pages_needing_review||[]).length;const message=review?'✓ 检查完成：'+report.completed_pages+'/'+report.total_pages+' 页齐全；有 '+review+' 页建议人工复核。':'✓ 检查完成：'+report.completed_pages+'/'+report.total_pages+' 页齐全，未发现需要人工复核的页面。';finishAction('audit',message,true)}else{finishAction('audit','检查失败：'+(data.detail||'请查看技术详情'),false)}}catch(error){result.textContent=String(error);finishAction('audit','无法检查，请确认本地服务仍在运行。',false)}}
function friendlyProofreadError(error){const text=String(error||'');if(text.includes('SSL')||text.includes('EOF')||text.includes('urlopen')||text.includes('ConnectError')||text.includes('NetworkError')||text.includes('ReadError'))return '与模型服务的网络连接中断';if(text.includes('401')||text.toLowerCase().includes('unauthorized'))return 'API Key 无效或没有权限';if(text.includes('404')||text.includes('Not Found'))return '模型地址或模型名称不正确';if(text.includes('429'))return '模型服务请求过多或账户额度不足';if(text.includes('用户已暂停'))return '用户已暂停';return '模型服务返回错误'}
function formatCountdown(iso){try{const diff=new Date(iso)-new Date();if(isNaN(diff)||diff<=0)return '即将重试';const s=Math.ceil(diff/1000);const m=Math.floor(s/60);const r=s%60;if(m>0)return m+'分'+(r>0?r+'秒':'');return r+'秒';}catch(e){return '即将重试'}}
function renderProofreadStatus(state){
    const status=document.querySelector('#proofread-status');
    const progress=document.querySelector('#proofread-progress');
    const button=document.querySelector('#proofread-button');
    const pauseBtn=document.querySelector('#proofread-pause-button');
    const resumeBtn=document.querySelector('#proofread-resume-button');
    const total=Number(state.total_pages||0);
    const page=Number(state.page||0);
    const done=Number(state.completed||0)+Number(state.skipped||0);
    pauseBtn.hidden=true;
    resumeBtn.hidden=true;
    if(state.status==='connecting'){progress.hidden=false;progress.max=total||1;progress.value=done;button.disabled=true;status.className='action-status';status.textContent='正在连接模型，准备从第 '+page+' 页继续；已保留 '+done+'/'+total+' 页。';setTimeout(proofreadStatus,2000)}
    else if(state.status==='waiting_network'){progress.hidden=false;progress.max=total||1;progress.value=done;button.disabled=true;pauseBtn.hidden=false;status.className='action-status';const countdown=state.next_retry_at?formatCountdown(state.next_retry_at):state.retry_in_seconds+'秒';status.textContent='网络波动，等待 '+countdown+' 后自动重试（第 '+state.retry_attempt+' 次）；已完成 '+done+'/'+total+' 页。';setTimeout(proofreadStatus,1000)}
    else if(state.status==='running'){progress.hidden=false;progress.max=total||1;progress.value=done;button.disabled=true;pauseBtn.hidden=false;status.className='action-status';status.textContent='正在处理第 '+page+' 页；已完成 '+done+'/'+total+' 页。页面会自动更新。';setTimeout(proofreadStatus,2000)}
    else if(state.status==='paused'){progress.hidden=false;progress.max=total||1;progress.value=done;button.disabled=true;resumeBtn.hidden=false;status.className='action-status';status.textContent='校对已暂停：将从第 '+page+' 页继续；已完成 '+done+'/'+total+' 页。';}
    else if(state.status==='complete'){progress.hidden=false;progress.max=total||1;progress.value=total||1;button.disabled=false;status.className='action-status ok';status.textContent='✓ 校对完成：'+total+'/'+total+' 页。结果已保存到输出文件夹。'}
    else if(state.status==='failed'){progress.hidden=false;progress.max=total||1;progress.value=done;button.disabled=false;resumeBtn.hidden=false;status.className='action-status error';status.textContent='校对已暂停：'+friendlyProofreadError(state.error)+'。已完成 '+done+'/'+total+' 页，停在第 '+page+' 页；点击“继续校对”从这里恢复。'}
    else{progress.hidden=true;button.disabled=false;status.className='action-status';status.textContent='尚未开始校对。'}}
async function proofread(){startAction('proofread','正在连接模型，请稍候…');try{const response=await fetch('/api/jobs/'+val('#job')+'/proofread',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({base_url:val('#base'),model:val('#model-name'),api_key:val('#key'),max_pages:optionalNumber('#proofread-max')})});const data=await show(response);if(response.ok&&data.status==='proofread_started'){setTimeout(proofreadStatus,500)}else{finishAction('proofread','无法开始校对：'+(data.detail||'请检查模型配置'),false)}}catch(error){result.textContent=String(error);finishAction('proofread','无法连接本地服务，请刷新页面后重试。',false)}}
async function pauseProofread(){try{const response=await fetch('/api/jobs/'+val('#job')+'/proofread-pause',{method:'POST'});const data=await show(response);if(response.ok){setTimeout(proofreadStatus,500)}else{result.textContent=data.detail||'无法暂停';}}catch(error){result.textContent=String(error);}}
async function resumeProofread(){try{const response=await fetch('/api/jobs/'+val('#job')+'/proofread-resume',{method:'POST'});const data=await show(response);if(response.ok&&data.status==='proofread_started'){setTimeout(proofreadStatus,500)}else if(response.ok){setTimeout(proofreadStatus,500)}else{result.textContent=data.detail||'无法继续';}}catch(error){result.textContent=String(error);}}
async function proofreadStatus(){try{const response=await fetch('/api/jobs/'+val('#job')+'/proofread-status');const state=await show(response);if(response.ok)renderProofreadStatus(state);else finishAction('proofread','无法读取校对状态。',false)}catch(error){result.textContent=String(error);finishAction('proofread','无法读取校对状态，请确认本地服务仍在运行。',false)}}
async function shutdown(){if(!confirm('停止服务后网页将不可用，是否继续？'))return;try{const response=await fetch('/api/shutdown',{method:'POST'});const data=await show(response);if(response.ok){document.body.innerHTML='<h1>BookOCR 服务已停止</h1><p>您可以关闭本页面和终端窗口。</p>';}else{result.textContent=data.detail||'无法停止服务';}}catch(error){result.textContent=String(error);}}
loadConfig();
</script></body></html>"""


def _choose_with_macos(script: str) -> str:
    """Open a native Finder chooser without sending a file anywhere."""
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("已取消选择" if "User canceled" in result.stderr else "无法打开 macOS 文件选择器")
    return result.stdout.strip()


def _local_port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def create_app(workspace: Path | None = None) -> FastAPI:
    workspace = (workspace or Path.cwd()).resolve()
    store = JobStore(workspace / "output" / "bookocr.sqlite3")
    settings = LlmSettingsStore()
    active_jobs: set[str] = set()
    active_lock = threading.Lock()
    pause_events: dict[str, threading.Event] = {}
    task_states: dict[str, dict[str, object]] = {}
    app = FastAPI(title="BookOCR", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.post("/api/shutdown")
    def shutdown() -> dict[str, str]:
        """Signal the server to shut down gracefully after returning the response."""
        def delayed_stop() -> None:
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=delayed_stop, daemon=True).start()
        return {"status": "shutting_down"}

    @app.get("/api/config/llm")
    def get_llm_settings() -> dict[str, str | bool]:
        return settings.load().__dict__

    @app.put("/api/config/llm")
    def save_llm_settings(request: SaveLlmSettingsRequest) -> dict[str, str | bool]:
        try:
            return settings.save(
                base_url=request.base_url,
                model=request.model,
                api_key=request.api_key or None,
                auto_proofread=request.auto_proofread,
            ).__dict__
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/doctor")
    def doctor() -> dict[str, str]:
        return {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version(), "workspace": str(workspace)}

    @app.post("/api/pick-pdf")
    def pick_pdf() -> dict[str, str]:
        try:
            path = _choose_with_macos('POSIX path of (choose file with prompt "选择扫描 PDF")')
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if Path(path).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="请选择 PDF 文件")
        return {"path": path}

    @app.post("/api/pick-output-dir")
    def pick_output_dir() -> dict[str, str]:
        try:
            return {"path": _choose_with_macos('POSIX path of (choose folder with prompt "选择输出目录")')}
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/mlx/start")
    def start_mlx() -> dict[str, str]:
        if _local_port_is_open(8118):
            return {"status": "already_running", "message": "Apple 加速服务已经在运行。"}
        launcher = workspace / "启动 MLX 加速服务.command"
        if not launcher.exists():
            raise HTTPException(status_code=500, detail="找不到“启动 MLX 加速服务.command”")
        try:
            subprocess.Popen(["open", str(launcher)])
        except OSError as exc:
            raise HTTPException(status_code=500, detail="无法启动 Apple 加速服务") from exc
        return {"status": "starting", "message": "已打开加速服务终端，请保持该窗口打开；就绪后可继续识别。"}

    @app.get("/api/mlx/status")
    def mlx_status() -> dict[str, bool]:
        return {"running": _local_port_is_open(8118)}

    @app.post("/api/jobs")
    def create_job(request: CreateJobRequest) -> dict:
        try:
            # Finder/Terminal often copies a path wrapped in matching quotes.
            # Treat those quotes as transport syntax, not part of the filename.
            pdf_text = request.pdf_path.strip().strip("\"'")
            job = store.create_job(Path(pdf_text).expanduser(), Path(request.output_dir), request.batch_size)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"id": job.id, "total_pages": job.total_pages, "batch_size": job.batch_size, "output_dir": str(job.output_dir)}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        try:
            job = store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        with active_lock:
            active = job.id in active_jobs
            ocr_status = dict(task_states.get(job.id, {}))
        pages = store.summary(job.id)
        if ocr_status.get("status") == "retrying_cpu":
            baseline = int(ocr_status["retry_baseline_complete"])
            ocr_status["retry_completed"] = str(max(0, pages.get("complete", 0) - baseline))
        elif pages.get("running", 0):
            # A retry may have been launched by the standalone recovery command
            # (or survive a web-service restart). Its second attempts still let
            # the UI report honest retry progress rather than a vague "running".
            retry = store.retry_summary(job.id)
            retry_total = sum(retry.values())
            if retry.get("running", 0):
                ocr_status = {
                    "stage": "ocr",
                    "status": "retrying_cpu",
                    "retry_total": str(retry_total),
                    "retry_completed": str(retry.get("complete", 0)),
                }
        return {
            "id": job.id,
            "total_pages": job.total_pages,
            "pages": pages,
            "active": active,
            "ocr_status": ocr_status,
        }

    @app.get("/api/latest-job")
    def latest_job() -> dict:
        job = store.latest_job()
        if job is None:
            return {}
        with active_lock:
            active = job.id in active_jobs
        return {"id": job.id, "total_pages": job.total_pages, "pages": store.summary(job.id), "active": active}

    @app.get("/api/jobs/{job_id}/audit")
    def audit_status(job_id: str) -> dict:
        try:
            job = store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        report_path = audit_job(store, job)
        return {"report": str(report_path), "data": json.loads(report_path.read_text(encoding="utf-8"))}

    @app.post("/api/jobs/{job_id}/assemble")
    def assemble_job(job_id: str) -> dict:
        try:
            job = store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        try:
            output = assemble_raw_markdown(store, job)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"markdown": str(output)}

    @app.post("/api/jobs/{job_id}/run")
    def run_job(job_id: str, request: RunJobRequest) -> dict:
        try:
            job = store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        with active_lock:
            if job_id in active_jobs:
                raise HTTPException(status_code=409, detail="该任务已在运行")
            active_jobs.add(job_id)
            task_states[job_id] = {"stage": "ocr", "status": "running"}

        def worker() -> None:
            try:
                run_ocr_job(
                    store,
                    job,
                    cache_dir=workspace / ".cache",
                    max_pages=request.max_pages,
                    vl_rec_backend=request.vl_backend,
                    vl_rec_server_url=request.vl_server_url,
                )
                failed_pages = store.summary(job.id).get("failed", 0)
                if request.cpu_fallback and failed_pages:
                    # MLX can reject a small number of unusually shaped pages.
                    # Retrying only failed rows through the default local backend
                    # preserves successful work and avoids an MLX-only dead end.
                    with active_lock:
                        task_states[job_id] = {
                            "stage": "ocr",
                            "status": "retrying_cpu",
                            "retry_total": str(failed_pages),
                            "retry_baseline_complete": str(store.summary(job.id).get("complete", 0)),
                        }
                    run_ocr_job(store, job, cache_dir=workspace / ".cache", retry_failed=True)
                with active_lock:
                    task_states[job_id] = {"stage": "ocr", "status": "complete"}
                if request.auto_proofread and store.summary(job.id).get("complete", 0) == job.total_pages:
                    try:
                        base_url, model, api_key = settings.resolve(base_url="", model="", api_key="")
                        with active_lock:
                            task_states[job_id] = {"stage": "proofread", "status": "running"}
                        summary = proofread_job(store, job, LlmProfile("custom", base_url, model, api_key))
                        with active_lock:
                            task_states[job_id] = {
                                "stage": "proofread",
                                "status": "complete",
                                "output": str(summary.output),
                                "completed": str(summary.completed),
                                "skipped": str(summary.skipped),
                            }
                    except Exception as exc:
                        # OCR itself succeeded. Report the optional next stage
                        # separately so a missing model configuration cannot hide it.
                        with active_lock:
                            task_states[job_id] = {
                                "stage": "proofread",
                                "status": "failed",
                                "error": str(exc),
                            }
                elif request.auto_proofread:
                    with active_lock:
                        task_states[job_id] = {
                            "stage": "proofread",
                            "status": "waiting_for_ocr",
                            "message": "OCR 尚未完成全书；完成后再次继续识别会自动校对。",
                        }
            except Exception as exc:
                with active_lock:
                    task_states[job_id] = {"stage": "ocr", "status": "failed", "error": str(exc)}
            finally:
                with active_lock:
                    active_jobs.discard(job_id)

        threading.Thread(target=worker, daemon=True).start()
        return {"id": job_id, "status": "started", "pages": store.summary(job_id)}

    @app.post("/api/jobs/{job_id}/proofread")
    def run_proofread(job_id: str, request: ProofreadRequest) -> dict:
        try:
            job = store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        try:
            base_url, model, api_key = settings.resolve(
                base_url=request.base_url,
                model=request.model,
                api_key=request.api_key,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with active_lock:
            if job_id in active_jobs:
                raise HTTPException(status_code=409, detail="该任务已在运行")
            active_jobs.add(job_id)
            pause_event = threading.Event()
            pause_events[job_id] = pause_event
            saved_pages = len(list((job.output_dir / "proofread_pages").glob("page_*.md")))
            initial_progress: dict[str, object] = {
                "stage": "proofread",
                "status": "connecting",
                "page": min(saved_pages + 1, job.total_pages),
                "total_pages": job.total_pages,
                "completed": 0,
                "skipped": saved_pages,
            }
            task_states[job_id] = initial_progress
        progress_path = job.output_dir / "proofread_progress.json"
        progress_path.write_text(
            json.dumps(initial_progress, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        def worker() -> None:
            try:
                summary = proofread_job(
                    store,
                    job,
                    LlmProfile("custom", base_url, model, api_key),
                    max_pages=request.max_pages,
                    pause_event=pause_event,
                )
                with active_lock:
                    task_states[job_id] = {
                        "stage": "proofread",
                        "status": "complete",
                        "output": str(summary.output),
                        "completed": str(summary.completed),
                        "skipped": str(summary.skipped),
                    }
            except ProofreadPaused:
                # Progress file has already been updated to "paused" by
                # proofread_job; keep the in-memory state consistent.
                progress_path = job.output_dir / "proofread_progress.json"
                progress: dict[str, object] = {}
                if progress_path.exists():
                    try:
                        progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        progress = {}
                with active_lock:
                    task_states[job_id] = {
                        "stage": "proofread",
                        "status": "paused",
                        **{key: str(value) for key, value in progress.items() if key != "error" and key != "status"},
                        "error": "用户已暂停校对",
                    }
            except Exception as exc:  # Return concise local error to the UI; never retain the API key.
                progress_path = job.output_dir / "proofread_progress.json"
                progress = {}
                if progress_path.exists():
                    try:
                        progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        progress = {}
                with active_lock:
                    task_states[job_id] = {
                        "stage": "proofread",
                        "status": "failed",
                        "error": str(exc),
                        **{key: str(value) for key, value in progress.items() if key != "error" and key != "status"},
                    }
            finally:
                with active_lock:
                    active_jobs.discard(job_id)
                    pause_events.pop(job_id, None)

        threading.Thread(target=worker, daemon=True).start()
        return {"id": job_id, "status": "proofread_started"}

    @app.post("/api/jobs/{job_id}/proofread-pause")
    def pause_proofread(job_id: str) -> dict:
        try:
            job = store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        with active_lock:
            if job_id not in active_jobs:
                raise HTTPException(status_code=409, detail="该校对任务未在运行")
            pause_events.get(job_id, threading.Event()).set()
        progress_path = job.output_dir / "proofread_progress.json"
        progress: dict[str, object] = {}
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                progress = {}
        progress["status"] = "paused"
        progress["error"] = "用户已暂停校对"
        progress_path.write_text(
            json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {"id": job_id, "status": "paused"}

    @app.post("/api/jobs/{job_id}/proofread-resume")
    def resume_proofread(job_id: str) -> dict:
        try:
            job = store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        with active_lock:
            # If a worker is still alive and paused, clear its event.
            event = pause_events.get(job_id)
            if event is not None:
                event.clear()
                return {"id": job_id, "status": "resumed"}
            # Otherwise start a fresh worker from the saved checkpoint state.
            if job_id in active_jobs:
                raise HTTPException(status_code=409, detail="该任务已在运行")
            active_jobs.add(job_id)
            pause_event = threading.Event()
            pause_events[job_id] = pause_event
            saved_pages = len(list((job.output_dir / "proofread_pages").glob("page_*.md")))
            initial_progress: dict[str, object] = {
                "stage": "proofread",
                "status": "connecting",
                "page": min(saved_pages + 1, job.total_pages),
                "total_pages": job.total_pages,
                "completed": 0,
                "skipped": saved_pages,
            }
            task_states[job_id] = initial_progress
        progress_path = job.output_dir / "proofread_progress.json"
        progress_path.write_text(
            json.dumps(initial_progress, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        def worker() -> None:
            try:
                base_url, model, api_key = settings.resolve(base_url="", model="", api_key="")
                summary = proofread_job(
                    store,
                    job,
                    LlmProfile("custom", base_url, model, api_key),
                    pause_event=pause_event,
                )
                with active_lock:
                    task_states[job_id] = {
                        "stage": "proofread",
                        "status": "complete",
                        "output": str(summary.output),
                        "completed": str(summary.completed),
                        "skipped": str(summary.skipped),
                    }
            except ProofreadPaused:
                progress_path = job.output_dir / "proofread_progress.json"
                progress = {}
                if progress_path.exists():
                    try:
                        progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        progress = {}
                with active_lock:
                    task_states[job_id] = {
                        "stage": "proofread",
                        "status": "paused",
                        **{key: str(value) for key, value in progress.items() if key != "error" and key != "status"},
                        "error": "用户已暂停校对",
                    }
            except Exception as exc:
                progress_path = job.output_dir / "proofread_progress.json"
                progress = {}
                if progress_path.exists():
                    try:
                        progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        progress = {}
                with active_lock:
                    task_states[job_id] = {
                        "stage": "proofread",
                        "status": "failed",
                        "error": str(exc),
                        **{key: str(value) for key, value in progress.items() if key != "error" and key != "status"},
                    }
            finally:
                with active_lock:
                    active_jobs.discard(job_id)
                    pause_events.pop(job_id, None)

        threading.Thread(target=worker, daemon=True).start()
        return {"id": job_id, "status": "proofread_started"}

    @app.get("/api/jobs/{job_id}/proofread-status")
    def proofread_status(job_id: str) -> dict[str, object]:
        with active_lock:
            in_memory = task_states.get(job_id)
            worker_active = job_id in active_jobs
        try:
            job = store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        progress_path = job.output_dir / "proofread_progress.json"
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                progress = {}
            # If the file says a worker is active but no worker is actually
            # running (e.g. after a server restart), surface it as a resumable
            # failure so the UI offers a "continue" button instead of a stale
            # "pause" button.
            if not worker_active and progress.get("status") in (
                "running",
                "connecting",
                "waiting_network",
            ):
                progress = {
                    **progress,
                    "status": "failed",
                    "error": "服务已重启或校对中断，请点击“继续校对”恢复",
                }
                progress_path.write_text(
                    json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            return progress
        if in_memory is not None:
            return in_memory
        return {"stage": "proofread", "status": "not_started"}

    return app


def serve(workspace: Path, port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn

    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{port}")
    uvicorn.run(create_app(workspace), host="127.0.0.1", port=port)
