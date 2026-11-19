"use client";

import { useState } from "react";
import {
  ApprovalRequiredPayload,
  StreamHandlers,
  submitErpApproval,
} from "@/lib/agentClient";

interface ErpApprovalCardProps {
  approval: ApprovalRequiredPayload;
  handlers: StreamHandlers;
  onResolved: (decision: "approved" | "rejected") => void;
}

export function ErpApprovalCard({
  approval,
  handlers,
  onResolved,
}: ErpApprovalCardProps) {
  const initialQty = approval.pending_order?.quantity ?? 1;
  const initialDiscount = approval.pending_order?.discount_percent ?? 0;
  const initialTotal = approval.pending_order?.total_vnd;

  const [quantity, setQuantity] = useState<number>(initialQty);
  const [discount, setDiscount] = useState<number>(initialDiscount);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Estimate updated total if initial total_vnd was provided
  const estimatedUnitBase =
    initialTotal && initialQty > 0 && initialDiscount < 100
      ? initialTotal / (initialQty * (1 - initialDiscount / 100))
      : approval.pending_order?.unit_price_vnd;

  const estimatedTotal =
    estimatedUnitBase !== undefined
      ? Math.round(estimatedUnitBase * quantity * (1 - discount / 100))
      : initialTotal;

  const handleDecision = async (action: "approve" | "reject") => {
    setSubmitting(true);
    setErrorMsg(null);

    await submitErpApproval(
      {
        thread_id: approval.thread_id,
        action,
        modified_order:
          action === "approve"
            ? {
                sku: approval.pending_order.sku,
                quantity,
                discount_percent: discount,
              }
            : undefined,
      },
      {
        ...handlers,
        onDone: () => {
          handlers.onDone?.();
          onResolved(action === "approve" ? "approved" : "rejected");
        },
        onError: (err) => {
          setErrorMsg(err.message);
          handlers.onError?.(err);
          setSubmitting(false);
        },
      }
    );
  };

  return (
    <div className="rounded-xl border border-amber-500/40 bg-amber-950/30 p-4 space-y-3 shadow-lg">
      <div className="flex items-start justify-between gap-3">
        <div>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/20 border border-amber-500/40 px-2.5 py-0.5 text-xs font-semibold text-amber-300">
            ⏸️ Human-in-the-Loop Checkpoint ({approval.next_node})
          </span>
          <p className="mt-1.5 text-sm font-medium text-amber-100">
            {approval.message}
          </p>
        </div>
        <span className="text-xs font-mono text-amber-400/80">
          Thread: {approval.thread_id}
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm pt-1">
        <div>
          <label className="block text-xs font-medium text-amber-200/80 mb-1">
            Mã SKU
          </label>
          <input
            disabled
            value={approval.pending_order?.sku ?? ""}
            className="w-full rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-1.5 text-slate-300 font-mono text-sm cursor-not-allowed"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-amber-200/80 mb-1">
            Số lượng (chỉnh sửa được)
          </label>
          <input
            type="number"
            min={1}
            disabled={submitting}
            value={quantity}
            onChange={(e) =>
              setQuantity(Math.max(1, Number(e.target.value) || 1))
            }
            className="w-full rounded-lg border border-amber-500/40 bg-slate-900 px-3 py-1.5 text-slate-100 text-sm focus:border-amber-400 focus:outline-none"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-amber-200/80 mb-1">
            Chiết khấu % (chỉnh sửa được)
          </label>
          <input
            type="number"
            min={0}
            max={100}
            step={0.5}
            disabled={submitting}
            value={discount}
            onChange={(e) =>
              setDiscount(
                Math.min(100, Math.max(0, Number(e.target.value) || 0))
              )
            }
            className="w-full rounded-lg border border-amber-500/40 bg-slate-900 px-3 py-1.5 text-slate-100 text-sm focus:border-amber-400 focus:outline-none"
          />
        </div>
      </div>

      {estimatedTotal !== undefined && (
        <div className="flex items-center justify-between rounded-lg bg-slate-900/70 border border-slate-800 px-3 py-2 text-xs">
          <span className="text-slate-400">Tổng thanh toán dự kiến:</span>
          <span className="font-semibold text-amber-300 text-sm">
            {estimatedTotal.toLocaleString("vi-VN")} VNĐ
          </span>
        </div>
      )}

      {errorMsg && (
        <div className="rounded-lg border border-rose-500/40 bg-rose-950/40 px-3 py-2 text-xs text-rose-300">
          ⚠️ {errorMsg}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2.5 pt-1">
        <button
          type="button"
          disabled={submitting}
          onClick={() => handleDecision("approve")}
          className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-4 py-2 text-xs font-semibold text-white hover:bg-emerald-500 disabled:opacity-50 transition shadow-sm"
        >
          {submitting ? "Đang xử lý..." : "✅ Phê duyệt & Trừ kho SQLite"}
        </button>
        <button
          type="button"
          disabled={submitting}
          onClick={() => handleDecision("reject")}
          className="inline-flex items-center gap-1.5 rounded-lg bg-rose-600 px-4 py-2 text-xs font-semibold text-white hover:bg-rose-500 disabled:opacity-50 transition shadow-sm"
        >
          {submitting ? "Đang xử lý..." : "✖️ Từ chối đơn hàng"}
        </button>
      </div>
    </div>
  );
}
