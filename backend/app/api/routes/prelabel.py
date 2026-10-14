"""M7 预标注接口。

定位：**预标注（pre-annotation）不是「自动标注」**。模型给出的框是伪标签，
漏检 = 缺标注、误检 = 错标注，必须人工复核后才能作为训练数据。
本接口只负责调度与把结果交回现成流水线；血缘（权重 / conf / 时间）会随
`dataset_card.json` 一起落盘，界面上也明确提示需要复核。

预标注是批量推理，耗时较长，因此改成「后台任务 + 轮询」而不是同步请求。

M7-04 复核 / 编辑：`/jobs/{id}/review/*` 让人逐张修正伪标签（增删框、改坐标与
类别、标记已确认）。改动只作用于内存里的 IR，导出时随血缘一起记录。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ... import services_prelabel
from ...schemas import (
    PrelabelAnnotationRequest,
    PrelabelAnnotationUpdate,
    PrelabelExportRequest,
    PrelabelExportResponse,
    PrelabelImageStatusRequest,
    PrelabelJobResponse,
    PrelabelJobsResponse,
    PrelabelRequest,
    PrelabelReviewPageResponse,
    PrelabelReviewSampleResponse,
    PrelabelReviewSummaryResponse,
    PrelabelSamplesResponse,
)
from ...services_infer import capability_report, discover_weights

router = APIRouter(prefix="/api/prelabel", tags=["prelabel"])


@router.get("/formats")
def list_formats() -> dict:
    """可推理格式与本机可用性（与 M6 共用一份能力探测）。"""
    return {"ok": True, "formats": capability_report()}


@router.get("/weights")
def list_weights() -> dict:
    """可选权重来源（模型库 + 导出产物），与 M6 实时验证一致。"""
    return {"weights": discover_weights()}


@router.post("/jobs", response_model=PrelabelJobResponse)
def start_job(req: PrelabelRequest) -> PrelabelJobResponse:
    """启动一次预标注（后台线程跑，立即返回任务）。"""
    try:
        job = services_prelabel.start_job(req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PrelabelJobResponse(job=job.snapshot())


@router.get("/jobs", response_model=PrelabelJobsResponse)
def list_jobs() -> PrelabelJobsResponse:
    """列出预标注任务（内存态）。"""
    return PrelabelJobsResponse(jobs=[j.snapshot() for j in services_prelabel.list_jobs()])


@router.get("/jobs/{job_id}", response_model=PrelabelJobResponse)
def get_job(job_id: str) -> PrelabelJobResponse:
    job = services_prelabel.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"预标注任务不存在: {job_id}")
    return PrelabelJobResponse(job=job.snapshot())


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    """删除任务（只清内存，不删已导出的数据集）。"""
    if not services_prelabel.remove_job(job_id):
        raise HTTPException(status_code=404, detail=f"预标注任务不存在: {job_id}")
    return {"ok": True, "removed": job_id}


@router.get("/jobs/{job_id}/samples", response_model=PrelabelSamplesResponse)
def get_samples(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(12, ge=1, le=200),
    filter: str = Query("all", description="all | pending | confirmed | with_boxes | empty | edited"),
) -> PrelabelSamplesResponse:
    """预览样本：带预测框的图片清单，供前端叠加显示。"""
    try:
        payload = services_prelabel.samples(job_id, offset=offset, limit=limit, filter_name=filter)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"预标注任务不存在: {job_id}") from None
    return PrelabelSamplesResponse(**payload)


@router.post("/jobs/{job_id}/export", response_model=PrelabelExportResponse)
def export_job(job_id: str, req: PrelabelExportRequest) -> PrelabelExportResponse:
    """把预标注结果导出为数据集（类别规范化 / 清洗 / 划分 / 导出全部复用现成的）。"""
    try:
        payload = services_prelabel.export_job(job_id, req)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"预标注任务不存在: {job_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = services_prelabel.get_job(job_id)
    return PrelabelExportResponse(
        out_dir=payload["out_dir"],
        report=payload["report"],
        job=job.snapshot() if job else {},
    )


# ---------------------------------------------------------------------------
# M7-04 复核 / 编辑
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/review/samples", response_model=PrelabelReviewPageResponse)
def get_review_samples(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(24, ge=1, le=200),
    filter: str = Query("all", description="all | pending | confirmed | with_boxes | empty | edited"),
) -> PrelabelReviewPageResponse:
    """复核用的样本页（含稳定 box_id 与复核状态）。"""
    try:
        payload = services_prelabel.review_samples(
            job_id, offset=offset, limit=limit, filter_name=filter
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"预标注任务不存在: {job_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PrelabelReviewPageResponse(**payload)


@router.get("/jobs/{job_id}/review/summary", response_model=PrelabelReviewSummaryResponse)
def get_review_summary(job_id: str) -> PrelabelReviewSummaryResponse:
    """复核进度统计（已确认 / 待复核 / 人改过 / 删过多少框）。"""
    try:
        summary = services_prelabel.review_summary_of(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"预标注任务不存在: {job_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PrelabelReviewSummaryResponse(summary=summary)


@router.post("/jobs/{job_id}/review/annotations", response_model=PrelabelReviewSampleResponse)
def add_review_annotation(
    job_id: str, req: PrelabelAnnotationRequest
) -> PrelabelReviewSampleResponse:
    """新增一条人工标注（人画的框，或整图类别）。"""
    try:
        payload = services_prelabel.add_review_annotation(
            job_id, req.image_uid, req.category, bbox=req.bbox, kind=req.kind
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PrelabelReviewSampleResponse(**payload)


@router.patch(
    "/jobs/{job_id}/review/annotations/{ann_id}", response_model=PrelabelReviewSampleResponse
)
def update_review_annotation(
    job_id: str, ann_id: str, req: PrelabelAnnotationUpdate
) -> PrelabelReviewSampleResponse:
    """修改一条标注的坐标 / 类别。"""
    try:
        payload = services_prelabel.update_review_annotation(
            job_id, ann_id, category=req.category, bbox=req.bbox
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PrelabelReviewSampleResponse(**payload)


@router.delete(
    "/jobs/{job_id}/review/annotations/{ann_id}", response_model=PrelabelReviewSampleResponse
)
def remove_review_annotation(job_id: str, ann_id: str) -> PrelabelReviewSampleResponse:
    """删除一条标注。"""
    try:
        payload = services_prelabel.remove_review_annotation(job_id, ann_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PrelabelReviewSampleResponse(**payload)


@router.post(
    "/jobs/{job_id}/review/images/{image_uid}/status",
    response_model=PrelabelReviewSampleResponse,
)
def set_review_status(
    job_id: str, image_uid: str, req: PrelabelImageStatusRequest
) -> PrelabelReviewSampleResponse:
    """标记某张图为「已确认」或「待复核」。"""
    try:
        payload = services_prelabel.set_review_status(job_id, image_uid, req.status)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PrelabelReviewSampleResponse(**payload)
