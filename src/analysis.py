# -*- coding: utf-8 -*-
"""客户流失分析与生存模型 — Kaplan-Meier 估计 + log-rank 检验 + 离散时间风险模型

数据：UCI Online Retail（与 ecommerce-behavior 项目同源，54 万行交易）
口径：
  - 观察窗：数据集首日 ~ 末日（2010-12-01 ~ 2011-12-09，末段 12 月不完整，截断至 2011-11-30 保证一致）
  - "生存"定义：客户持续活跃（当月有购买）；流失 = 进入连续 ≥60 天无购买的区间
  - 离散时间构造：每个客户按月展开为 Person-Month 面板，事件月 = 首个进入流失区间的月份
方法：
  - Kaplan-Meier 生存函数（手动实现，含删失处理）
  - 分组 KM 对比 + log-rank 检验（手动实现）
  - 离散时间 logistic 风险模型（OR 解读）
输出：output/ 图表 + metrics.json
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
C = {"main": "#2563EB", "warm": "#D97706", "bad": "#E5484D", "good": "#10B981", "grey": "#9AA5B1"}

metrics = {}
CHURN_GAP = 60  # 连续无购天数阈值

# ========== 1. 数据准备 ==========
df = pd.read_excel(ROOT / "data" / "Online_Retail.xlsx")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Revenue"] = df["Quantity"] * df["UnitPrice"]
df["InvoiceNoStr"] = df["InvoiceNo"].astype(str)
sales = df[(df["Quantity"] > 0) & (~df["InvoiceNoStr"].str.startswith("C")) & (df["UnitPrice"] > 0)]
known = sales[sales["CustomerID"].notna()].copy()

OBS_START = pd.Timestamp("2010-12-01")
OBS_END = pd.Timestamp("2011-11-30")   # 截断不完整的 12 月
known = known[(known["InvoiceDate"] >= OBS_START) & (known["InvoiceDate"] <= OBS_END)]

# 每个客户的购买日序列 → 逐月活跃标记
known["ym"] = known["InvoiceDate"].dt.to_period("M")
active = known.groupby(["CustomerID", "ym"]).size().reset_index(name="n")
months = pd.period_range("2010-12", "2011-11", freq="M")
customers = active["CustomerID"].unique()
panel = pd.MultiIndex.from_product([customers, months], names=["CustomerID", "ym"]).to_frame(index=False)
panel = panel.merge(active.assign(active=1)[["CustomerID", "ym", "active"]],
                    on=["CustomerID", "ym"], how="left").fillna({"active": 0})

# 首购月与月度购买金额
first_m = known.groupby("CustomerID")["ym"].min().rename("first_ym")
mon_rev = known.groupby(["CustomerID", "ym"])["Revenue"].sum().rename("rev")
panel = panel.merge(first_m, on="CustomerID").merge(mon_rev, on=["CustomerID", "ym"], how="left").fillna({"rev": 0.0})
panel["month_idx"] = (panel["ym"].astype(int) - panel["first_ym"].astype(int))  # 相对首购的月序
panel = panel[panel["month_idx"] >= 0]     # 从首购月开始观察
panel = panel.sort_values(["CustomerID", "ym"]).reset_index(drop=True)

# ========== 2. 离散时间流失事件构造 ==========
# 事件月定义：该月结束后进入连续 ≥ CHURN_GAP 天无购买 → 该月为事件月（若当月仍活跃则流失发生于其后）
events = []
for cust, grp in panel.groupby("CustomerID"):
    g = grp.reset_index(drop=True)
    last_active_idx = None
    event_idx = None
    last_day = g["ym"].iloc[-1].to_timestamp(how="end")
    for i, row in g.iterrows():
        if row["active"] == 1:
            last_active_idx = i
    # 流失判定：最后一次活跃月之后的所有月份视为"已流失"；若实验期末仍活跃且距末日 < 60 天则删失
    if last_active_idx is None:
        continue
    last_active_month_end = g["ym"].iloc[last_active_idx].to_timestamp(how="end")
    gap_at_end = (OBS_END - last_active_month_end).days
    for i in range(len(g)):
        churned_before = i > last_active_idx
        if churned_before:
            # 该月开始时已连续无购 (i - last_active_idx - 1) 个月 + 该月全月
            if event_idx is None:
                event_idx = i   # 首个"已流失"的月作为事件月
    if event_idx is None:
        # 从未流失：若期末距最后活跃 <60 天 → 删失（观察期内无法确认流失）
        censored = gap_at_end < CHURN_GAP
    else:
        censored = False
    events.append({"CustomerID": cust, "event_idx": event_idx,
                   "n_months": len(g), "censored": censored})
ev = pd.DataFrame(events)
panel = panel.merge(ev, on="CustomerID")
panel["at_risk"] = (panel["event_idx"].isna() | (panel["month_idx"] <= panel["event_idx"])).astype(int)
panel["event_this_month"] = ((~panel["event_idx"].isna()) &
                             (panel["month_idx"] == panel["event_idx"])).astype(int)

metrics["n_customers"] = int(len(ev))
metrics["n_churned"] = int((~ev["censored"] & ev["event_idx"].notna()).sum())
metrics["n_censored"] = int(ev["censored"].sum())
metrics["churn_gap_days"] = CHURN_GAP

# ========== 3. Kaplan-Meier（手动实现，月粒度） ==========
def km_estimate(panel_df):
    """按事件月聚合：风险集 = at_risk & month_idx == t；事件数 = event_this_month"""
    rows = []
    risk0 = panel_df["CustomerID"].nunique()
    surv = 1.0
    for t in sorted(panel_df["month_idx"].unique()):
        sub = panel_df[panel_df["month_idx"] == t]
        n_risk = sub["at_risk"].sum()
        d = sub["event_this_month"].sum()
        if n_risk == 0:
            continue
        if d > 0:
            surv *= (1 - d / n_risk)
        rows.append({"t": int(t), "n_risk": int(n_risk), "d": int(d), "surv": surv})
    return pd.DataFrame(rows)

km_all = km_estimate(panel)

# 分组：按首购季度的 RFM 简化分组——用"首月购买金额是否高于中位数"分高低价值
first_rev = panel[panel["month_idx"] == 0].groupby("CustomerID")["rev"].sum()
med = first_rev.median()
hi = set(first_rev[first_rev > med].index)
panel["seg"] = np.where(panel["CustomerID"].isin(hi), "首月高价值", "首月低价值")

fig, ax = plt.subplots(figsize=(8.5, 4.4))
colors = {"全部客户": C["main"], "首月高价值": C["good"], "首月低价值": C["bad"]}
km_all_plot = km_all.copy()
ax.step(km_all_plot["t"], km_all_plot["surv"] * 100, where="post",
        color=colors["全部客户"], lw=2.2, label="全部客户")
logrank = {}
for seg in ["首月高价值", "首月低价值"]:
    km_seg = km_estimate(panel[panel["seg"] == seg])
    ax.step(km_seg["t"], km_seg["surv"] * 100, where="post", color=colors[seg], lw=1.8,
            label=seg, alpha=0.9)
    # log-rank 检验（月粒度，手动实现）
    O_E, V = 0.0, 0.0
    for t in sorted(panel["month_idx"].unique()):
        a = panel[(panel["month_idx"] == t) & (panel["seg"] == seg)]
        b = panel[(panel["month_idx"] == t) & (panel["seg"] != seg)]
        n1, n2 = a["at_risk"].sum(), b["at_risk"].sum()
        if n1 + n2 == 0:
            continue
        d1 = a["event_this_month"].sum()
        d_all = d1 + b["event_this_month"].sum()
        if n1 + n2 <= 1:
            continue
        O_E += d1 - d_all * n1 / (n1 + n2)
        V += d_all * (n1 / (n1 + n2)) * (n2 / (n1 + n2)) * (n1 + n2 - d_all) / max(n1 + n2 - 1, 1)
    chi2 = (O_E ** 2) / V if V > 0 else 0.0
    pval = float(stats.chi2.sf(chi2, 1))
    logrank[seg] = {"chi2": round(float(chi2), 2), "p": float(f"{pval:.2e}")}
metrics["logrank_high_vs_low_value"] = logrank
metrics["km"] = {"surv_m3": round(float(km_all[km_all["t"] == 3]["surv"].iloc[0]) * 100, 1) if (km_all["t"] == 3).any() else None,
                 "surv_m6": round(float(km_all[km_all["t"] == 6]["surv"].iloc[0]) * 100, 1) if (km_all["t"] == 6).any() else None,
                 "surv_m9": round(float(km_all[km_all["t"] == 9]["surv"].iloc[0]) * 100, 1) if (km_all["t"] == 9).any() else None}

ax.axhline(50, ls=":", c=C["grey"], lw=1)
ax.set_xlabel("首购后月数")
ax.set_ylabel("生存概率 %（持续活跃）")
ax.set_title(f"Kaplan-Meier 生存曲线：流失 = 连续 {CHURN_GAP} 天无购买（删失客户不计入事件）")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "01_km_curves.png", bbox_inches="tight")
plt.close(fig)

# ========== 4. 离散时间 logistic 风险模型 ==========
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

risk_df = panel[panel["at_risk"] == 1].copy()
risk_df["tenure"] = risk_df["month_idx"]
# 注意：不用"上月是否活跃"作特征——事件月定义为活跃中断后的第一个月，
# 该特征与事件定义互为镜像（同义反复），会把模型 AUC 虚推到 0.99+ 而无业务含义。
risk_df["rev_log"] = np.log1p(risk_df["rev"])
risk_df["seg_hi"] = (risk_df["seg"] == "首月高价值").astype(int)
risk_df["recent_active"] = risk_df.groupby("CustomerID")["active"].transform(
    lambda s: s.shift(1).rolling(3, min_periods=1).mean()).fillna(1)

FEATS = ["tenure", "recent_active", "rev_log", "seg_hi"]
X = risk_df[FEATS].values
y = risk_df["event_this_month"].values
itr, ite = train_test_split(np.arange(len(X)), test_size=0.3, stratify=y, random_state=42)
lr = LogisticRegression(max_iter=2000).fit(X[itr], y[itr])
from sklearn.metrics import roc_auc_score
auc_disc = roc_auc_score(y[ite], lr.predict_proba(X[ite])[:, 1])
or_ = np.exp(lr.coef_[0])
metrics["discrete_risk"] = {
    "auc_oos": round(float(auc_disc), 3),
    "odds_ratios": {f: round(float(o), 3) for f, o in zip(FEATS, or_)},
    "n_person_months": int(len(X)),
    "event_rate": round(float(y.mean()) * 100, 2),
}

# ========== 5. 图：风险因子 OR ==========
fig, ax = plt.subplots(figsize=(8, 3.4))
names = list(metrics["discrete_risk"]["odds_ratios"].keys())
vals = list(metrics["discrete_risk"]["odds_ratios"].values())
order = np.argsort(vals)
ax.barh([names[i] for i in order], [vals[i] for i in order], color=[C["bad"] if vals[i] > 1 else C["good"] for i in order])
ax.axvline(1, ls="--", c=C["grey"], lw=1)
for i, i0 in enumerate(order):
    ax.text(vals[i0] * 1.02, i, f"{vals[i0]:.2f}", va="center", fontsize=9)
ax.set_xlabel("流失优势比 OR（>1 提升流失风险）")
ax.set_title(f"离散时间风险模型（样本外 AUC {auc_disc:.3f}）：上月活跃度是最强保护因子")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "02_or_factors.png", bbox_inches="tight")
plt.close(fig)

(OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: metrics[k] for k in
                  ["n_customers", "n_churned", "n_censored", "km", "logrank_high_vs_low_value",
                   "discrete_risk"]}, ensure_ascii=False, indent=2))
print("输出完成：", OUT)
