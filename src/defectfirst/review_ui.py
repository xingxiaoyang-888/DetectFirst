"""Offline review sheet; explicit answers only, no model predictions or auto pass."""

from __future__ import annotations

import base64
import html
import json
import shutil
from pathlib import Path

from defectfirst.controls.artifacts import verify_artifact
from defectfirst.controls.review import CHECKS, requires_second
from defectfirst.io import atomic_bytes, read_jsonl, within, write_json


def export_review(config: dict, root: Path, output: Path) -> dict:
    groups = read_jsonl(within(root, config["candidates"]))
    offset, limit = config.get("offset", 0), config.get("limit", 40)
    groups = groups[offset : offset + limit]
    if not groups:
        raise ValueError("No candidate groups in the requested review batch")
    output.mkdir(parents=True, exist_ok=True)
    cards, metadata = [], []
    for index, group in enumerate(groups):
        verify_artifact(root, group)
        info = {
            "group_id": group["group_id"],
            "content_sha256": group["content_sha256"],
            "conditions": group["condition_ids"],
            "second_required": requires_second(group, config.get("second_fraction", 0.2)),
        }
        metadata.append(info)
        candidate = within(root, group["files"]["M"]).parent
        image_name = f"group_{index}.png"
        shutil.copy2(candidate / "review.png", output / image_name)
        native_images = []
        for name in ("normal", "defect"):
            source = within(root, group["primitive"][name])
            destination = f"group_{index}_{name}.png"
            shutil.copy2(source, output / destination)
            native_images.append(
                f'<figure><img src="{destination}" alt="native {name}"><figcaption>Native {name}</figcaption></figure>'
            )
        mask_dest = f"group_{index}_M.png"
        shutil.copy2(within(root, group["files"]["M"]), output / mask_dest)
        form = []
        for condition in group["condition_ids"]:
            cells = []
            for check in sorted(CHECKS):
                if condition == 0 and check == "editing_effective":
                    cells.append("<td>Original: N/A</td>")
                else:
                    cells.append(
                        f'<td><select data-group="{index}" data-condition="{condition}" data-check="{check}"><option value="">Unanswered</option><option value="true">Pass</option><option value="false">Fail</option></select></td>'
                    )
            form.append(f"<tr><th>{condition}</th>{''.join(cells)}</tr>")
        cards.append(f'''<article id="card{index}"><h2>{html.escape(group["group_id"])}</h2>
<p>Second independent review required by sampling/diagnosis rule: {info["second_required"]}</p>
<img class="mosaic" src="{image_name}" alt="Crossed states with M red and G cyan">
<details><summary>Open native normal / defect at full resolution</summary><div class="native">{"".join(native_images)}</div><a href="{mask_dest}" download>Download current M</a></details>
<p>If M needs editing, revise its PNG in an image annotation tool, rebuild into a new revision, and export a new sheet. This sheet cannot silently approve a changed mask.</p>
<table><thead><tr><th>Condition</th>{"".join(f"<th>{name}</th>" for name in sorted(CHECKS))}</tr></thead><tbody>{"".join(form)}</tbody></table>
<label>Decision <select id="decision{index}"><option value="">Unanswered</option><option>accept</option><option>reject</option><option>uncertain</option><option>corrected</option></select></label>
<label>Active review seconds <input id="seconds{index}" type="number" min="0" placeholder="Measured active time"></label>
<label>Reason <input id="reason{index}" size="65"></label></article>''')
    # Encode metadata as base64 to avoid closing-script injection from filenames / IDs.
    encoded = base64.b64encode(json.dumps(metadata, ensure_ascii=False).encode()).decode()
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>DefectFirst review</title>
<style>body{font:16px system-ui;background:#f1f5f9;color:#172033;margin:32px}article{background:white;padding:24px;margin:24px 0;border-radius:12px}table{border-collapse:collapse;margin:20px 0}td,th{padding:8px;border:1px solid #ccc}input,select,button{padding:8px;margin:6px}.mosaic{max-width:100%}.native{display:flex;gap:20px}.native img{max-width:100%}figure{max-width:48%;margin:0}header{background:#e0f2fe;padding:16px}button{cursor:pointer}</style>
<header><h1>DefectFirst · Human review</h1><p>One mask per defect core. Verify native defect and each condition. No answers are preselected. Red: M; cyan: G. Do not view detector predictions.</p><label>Reviewer ID <input id="reviewer"></label><button id="save">Export completed reviews (JSONL)</button><p id="message"></p></header>
"""
        + "".join(cards)
        + '''<script>
const metadata = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob("'''
        + encoded
        + """"), c => c.charCodeAt(0))));
document.getElementById('save').onclick = () => {
 const reviewer = document.getElementById('reviewer').value.trim();
 if (!reviewer) { alert('Enter the reviewer ID'); return; }
 const records = [];
 for (let i=0;i<metadata.length;i++) {
  const decision = document.getElementById('decision'+i).value;
  if (!decision) continue;
  const secondsText=document.getElementById('seconds'+i).value;
  if (!secondsText || !Number.isFinite(Number(secondsText)) || Number(secondsText)<0) { alert('Record active review time for group '+metadata[i].group_id); return; }
  const conditions={};
  for (const e of metadata[i].conditions) {
   conditions[e]={};
   for (const check of ['boundary_valid','defect_preserved','editing_effective','mask_correct','normal_valid']) {
    if (e===0 && check==='editing_effective') { conditions[e][check]=true; continue; }
    const field=document.querySelector(`[data-group="${i}"][data-condition="${e}"][data-check="${check}"]`);
    if (field.value==='') { alert('Complete every condition check for '+metadata[i].group_id); return; }
    conditions[e][check]=field.value==='true';
   }
  }
  records.push({group_id:metadata[i].group_id,content_sha256:metadata[i].content_sha256,reviewer,seconds:Number(secondsText),decision,conditions,reason:document.getElementById('reason'+i).value,reviewed_at:new Date().toISOString()});
 }
 if (!records.length) { alert('No completed reviews'); return; }
 const blob=new Blob([records.map(r=>JSON.stringify(r)).join('\\n')+'\\n'],{type:'application/jsonl'});
 const link=document.createElement('a'); link.href=URL.createObjectURL(blob);link.download='reviews.jsonl';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
 document.getElementById('message').textContent='Exported '+records.length+' explicit reviews. Keep each reviewer file separately before merging.';
};
</script></html>"""
    )
    atomic_bytes(output / "index.html", page.encode("utf-8"))
    write_json(output / "review_metadata.json", metadata)
    return {"status": "WAITING_HUMAN", "groups": len(groups), "entry": str(output / "index.html")}
