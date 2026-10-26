# -*- coding: utf-8 -*-
"""生成 churn-survival 在线报告页。用法：python3 src/build_report.py"""
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
PUB = ROOT / "public"
(CHARTS := PUB / "charts").mkdir(parents=True, exist_ok=True)

M = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
for png in OUT.glob("*.png"):
    shutil.copy(png, CHARTS / png.name)


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def metric_card(v, label, note=""):
    return f'<div class="kpi"><b>{v}</b><span>{label}</span><em>{note}</em></div>'


lr = M["logrank_high_vs_low_value"]["首月高价值"]
dr = M["discrete_risk"]
or_rows = "".join(
    f"<tr><td>{esc(k)}</td><td>{v:.3f}</td><td>{'风险因子' if v > 1 else '保护因子'}</td></tr>"
    for k, v in dr["odds_ratios"].items()
)

kpi = (
    metric_card(f"{M['n_customers']:,}", "观察客户", f"流失 {M['n_churned']:,} · 删失 {M['n_censored']:,}")
    + metric_card(f"{M['km']['surv_m3']}%", "3月生存率", "持续活跃的客户占比")
    + metric_card(f"{dr['auc_oos']:.3f}", "风险模型 AUC", "样本外（30% 留出）")
    + metric_card(f"{lr['p']:.0e}", "log-rank p 值", "高低价值组生存差异")
)

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>客户流失分析与生存模型 · Kaplan-Meier + 离散时间风险模型</title>
  <meta property="og:title" content="客户流失分析与生存模型 · Kaplan-Meier + 离散时间风险模型">
  <meta property="og:description" content="">
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://cloud1-d4gsgj705497df85d-1501559031.tcloudbaseapp.com/churn/">

<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='22' fill='%23B4540A'/><text x='50' y='68' font-size='44' text-anchor='middle' fill='white' font-family='sans-serif' font-weight='bold'>生</text></svg>">
<style>
:root {{ --bg:#F8F6F3; --card:#fff; --text:#26211B; --muted:#8A8378; --border:#EAE5DD; --accent:#B4540A; }}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text); line-height:1.75; }}
header {{ background:linear-gradient(135deg,#8F4408,#B4540A); color:#fff; padding:26px 22px 20px; border-radius:0 0 22px 22px; }}
header h1 {{ font-size:22px; }} header p {{ font-size:13px; opacity:.9; }}
main {{ max-width:900px; margin:0 auto; padding:18px 16px 50px; }}
.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:-6px; }}
@media (max-width:640px) {{ .kpis {{ grid-template-columns:repeat(2,1fr); }} }}
.kpi {{ background:var(--card); border-radius:14px; box-shadow:0 2px 10px rgba(38,33,27,.07); padding:14px 10px; text-align:center; margin-top:14px; }}
.kpi b {{ display:block; font-size:22px; color:var(--accent); }}
.kpi span {{ font-size:12px; font-weight:700; }} .kpi em {{ display:block; font-size:11px; font-style:normal; color:var(--muted); }}
section {{ margin-top:22px; }}
.card {{ background:var(--card); border-radius:14px; box-shadow:0 2px 10px rgba(38,33,27,.07); padding:18px 20px; margin-top:12px; }}
h2 {{ font-size:17px; border-left:4px solid var(--accent); padding-left:10px; }}
p {{ font-size:14px; margin-top:8px; }}
ul {{ font-size:14px; padding-left:20px; }} li {{ margin-top:4px; }}
img {{ max-width:100%; border-radius:10px; border:1px solid var(--border); margin-top:10px; }}
.figcap {{ font-size:12px; color:var(--muted); margin-top:4px; }}
table.tbl {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }}
.tbl th, .tbl td {{ border-bottom:1px solid var(--border); padding:7px 10px; text-align:left; }}
.tbl th {{ color:var(--muted); font-weight:600; }}
.note {{ background:#FBF2E8; border:1px solid #EFDCC3; border-radius:10px; padding:10px 14px; font-size:13px; margin-top:10px; }}
footer {{ text-align:center; font-size:12px; color:var(--muted); padding:26px 0 34px; }}
</style>
</head>
<body>
<header>
  <h1>客户流失分析与生存模型</h1>
  <p>UCI Online Retail · Kaplan-Meier 估计（手动实现）+ log-rank 检验 + 离散时间 logistic 风险模型 · Python</p>
</header>
<main>

<div class="kpis">{kpi}</div>

<section>
  <h2>1 · 为什么用生存分析研究流失</h2>
  <div class="card">
    <p>流失建模的经典错误是把它当普通二分类："观察期末是否活跃"把删失信息全部扔掉——观察期末仍活跃的客户可能只是"还没到流失时间"，把他们当负样本会系统性低估流失率。<b>生存分析正是处理这类右删失数据的工具</b>：每个客户有观察时长和事件指示，未流失的按删失处理。</p>
    <p>口径：观察窗 2010-12 ~ 2011-11（截掉不完整的 12 月）；流失 = 连续 {M['churn_gap_days']} 天无购买；月粒度离散化。{M['n_customers']:,} 名客户中 {M['n_churned']:,} 名在观察期内确认流失，{M['n_censored']:,} 名按删失处理。</p>
  </div>
</section>

<section>
  <h2>2 · Kaplan-Meier 生存曲线</h2>
  <div class="card">
    <img src="charts/01_km_curves.png" alt="KM曲线">
    <p><b>方法：</b>KM 估计器 S(t) = ∏(1 − dₜ/nₜ)，逐月计算风险集与事件数，删失客户在退出后不再计入风险集。全部为手动实现（NumPy），无生存分析库依赖。</p>
    <p><b>结论：</b>3 月生存率 {M['km']['surv_m3']}%、6 月 {M['km']['surv_m6']}%、9 月 {M['km']['surv_m9']}%。首月的大幅下降来自约三分之一的一次性采购客户——这是 B2B 批发的真实结构，也是客户分层（RFM）必要性的直接证据：一次性买家与长期客户根本不该用同一套留存策略。</p>
  </div>
</section>

<section>
  <h2>3 · log-rank 检验：高价值客户真的活得更久吗</h2>
  <div class="card">
    <p>按首月购买金额中位数把客户分为「首月高价值 / 低价值」两组，log-rank 检验统计量 χ² = {lr['chi2']}，<b>p = {lr['p']:.1e}</b>——两组生存曲线的差异极显著，方向上高价值组全程位于上方（流失更慢）。</p>
    <p><b>业务含义：</b>首月消费金额是早期识别留存潜力的信号——新客首单的规模值得投入运营资源（首单关怀、组合推荐），这与 Cohort 留存矩阵的结论互相印证。</p>
  </div>
</section>

<section>
  <h2>4 · 离散时间风险模型：谁在什么时候流失</h2>
  <div class="card">
    <img src="charts/02_or_factors.png" alt="风险因子">
    <p><b>方法：</b>把每个客户按月展开为 Person-Month 面板（{dr['n_person_months']:,} 条记录，月流失率 {dr['event_rate']}%），以"该月是否确认流失"为因变量拟合 logistic 回归，系数取幂得优势比 OR。样本外（30% 留出）AUC = <b>{dr['auc_oos']:.3f}</b>。</p>
    <table class="tbl">
      <thead><tr><th>特征</th><th>OR</th><th>解读</th></tr></thead>
      <tbody>{or_rows}</tbody>
    </table>
    <p><b>结论：</b>① 近三个月活跃占比是最强的先兆信号（活跃度下降 → 流失几率倍增），是流失预警的核心监控变量；② 消费金额是强保护因子；③ 高价值客户流失风险低约 26%。</p>
    <div class="note">实现备注：模型刻意不使用"上月是否活跃"作特征——它与本项目的流失事件定义互为镜像（同义反复），会把 AUC 虚推到 0.99 以上而毫无业务含义。去掉之后 AUC 仍有 0.903，且所有 OR 都可业务解读。</div>
  </div>
</section>

<section>
  <h2>5 · 局限与改进方向</h2>
  <div class="card">
    <ul>
      <li>流失阈值（{M['churn_gap_days']} 天）是业务约定而非统计估计；可用购区间分布（如帕累托/NBD）反推，或对阈值做敏感性分析。</li>
      <li>离散时间 logistic 是 Cox 比例风险模型的分组近似；基线风险函数未随时间灵活化，可加入时间哑变量放松比例风险假设。</li>
      <li>删失客户的"最后活跃时点"之后即停止贡献风险信息；更精细的实现应把删失月也纳入风险集至其退出时点。</li>
      <li>可扩展：与 Pareto/NBD、BG/NBD 概率模型对比 CLV 预测；加入购物篮多样性等行为特征。</li>
    </ul>
  </div>
</section>

<footer>数据来源：UCI Online Retail（同电商行为分析项目）· 全部手动实现，无生存分析库依赖 · 代码见仓库 · <a href="/">← 返回作品集主页</a></footer>
</main>
</body>
</html>"""

(PUB / "index.html").write_text(html, encoding="utf-8")
print(f"报告已生成: {PUB / 'index.html'}")
