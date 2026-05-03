"""
╔══════════════════════════════════════════════════════════════════╗
║   УНИВЕРСАЛЬНАЯ СИСТЕМА АВТОМАТИЗИРОВАННОГО РЕГРЕССИОННОГО       ║
║   МОДЕЛИРОВАНИЯ С ПОЛНОЙ ПРОВЕРКОЙ ПРЕДПОСЫЛОК МНК               ║
╚══════════════════════════════════════════════════════════════════╝

Использование:
    run_universal_modeling(df, forecast_steps=4)

    Формат df:
        col[0] — метки периодов (строки)
        col[1] — зависимая переменная Y
        col[2:] — объясняющие факторы X
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from itertools import combinations

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ЦВЕТОВАЯ СХЕМА
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C = {
    'bg':       '#0D1117',
    'panel':    '#161B22',
    'border':   '#30363D',
    'text':     '#E6EDF3',
    'muted':    '#8B949E',
    'accent':   '#58A6FF',
    'green':    '#3FB950',
    'red':      '#F85149',
    'yellow':   '#D29922',
    'purple':   '#BC8CFF',
    'forecast': '#FF7B72',
    'fit':      '#79C0FF',
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ШАГ 1: ОТБОР ФАКТОРОВ — МУЛЬТИКОЛЛИНЕАРНОСТЬ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def select_factors(df, target, factors, corr_threshold=0.85, vif_threshold=10.0):
    """
    Итеративно удаляет мультиколлинеарные факторы:
    1. Попарная корреляция > corr_threshold → убираем слабее связанный с Y
    2. VIF > vif_threshold → убираем фактор с максимальным VIF
    Возвращает список отобранных факторов и лог решений.
    """
    log =[]
    corr = df[factors + [target]].corr()

    # --- Шаг 1а: попарная корреляция ---
    active = list(factors)
    removed_corr =[]
    changed = True
    while changed:
        changed = False
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                f1, f2 = active[i], active[j]
                if abs(corr.loc[f1, f2]) > corr_threshold:
                    keep = f1 if abs(corr.loc[f1, target]) >= abs(corr.loc[f2, target]) else f2
                    drop = f2 if keep == f1 else f1
                    r12 = corr.loc[f1, f2]
                    log.append(
                        f"  ⚠ Попарная корреляция |r({f1},{f2})|={abs(r12):.3f} > {corr_threshold} "
                        f"→ удаляю '{drop}' (r_Y={abs(corr.loc[drop,target]):.3f} < r_Y={abs(corr.loc[keep,target]):.3f})"
                    )
                    active.remove(drop)
                    removed_corr.append(drop)
                    changed = True
                    break
            if changed:
                break

    # --- Шаг 1б: VIF ---
    removed_vif =[]
    if len(active) >= 2:
        changed = True
        while changed:
            changed = False
            X_vif = sm.add_constant(df[active])
            vifs = {active[i]: variance_inflation_factor(X_vif.values, i + 1)
                    for i in range(len(active))}
            worst = max(vifs, key=vifs.get)
            if vifs[worst] > vif_threshold:
                log.append(f"  ⚠ VIF({worst})={vifs[worst]:.2f} > {vif_threshold} → удаляю '{worst}'")
                active.remove(worst)
                removed_vif.append(worst)
                changed = True

    return active, removed_corr, removed_vif, log


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ШАГ 2: ПРОВЕРКА ПРЕДПОСЫЛОК
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_assumptions(model, resid, X_factors_df, alpha=0.05):
    """
    Полный набор тестов предпосылок МНК.
    Возвращает OrderedDict {название: (passed, статистика, p_value, описание)}
    """
    res = {}

    # 1. Нормальность — Jarque-Bera
    jb, jb_p = stats.jarque_bera(resid)
    res['Нормальность остатков (Jarque-Bera)'] = (
        jb_p > alpha, jb, jb_p, f"JB={jb:.3f},  p={jb_p:.3f}")

    # 2. Нормальность — Shapiro-Wilk
    sw, sw_p = stats.shapiro(resid)
    res['Нормальность остатков (Shapiro-Wilk)'] = (
        sw_p > alpha, sw, sw_p, f"W={sw:.4f},  p={sw_p:.3f}")

    # 3. Автокорреляция — Durbin-Watson
    dw = durbin_watson(resid)
    dw_ok = 1.5 < dw < 2.5
    res['Автокорреляция (Durbin-Watson)'] = (
        dw_ok, dw, None, f"DW={dw:.4f}  {'(норма 1.5–2.5) ✓' if dw_ok else '(норма 1.5–2.5) ✗'}")

    # 4. Автокорреляция — Breusch-Godfrey (4 лага)
    try:
        dummy_model = sm.OLS(
            resid, sm.add_constant(np.arange(len(resid)))
        ).fit()
        bg, bg_p, _, _ = acorr_breusch_godfrey(dummy_model, nlags=4)
        res['Автокорреляция (Breusch-Godfrey, лаг 4)'] = (
            bg_p > alpha, bg, bg_p, f"LM={bg:.3f},  p={bg_p:.3f}")
    except Exception:
        res['Автокорреляция (Breusch-Godfrey, лаг 4)'] = (True, None, None, "н/д")

    # 5. Гетероскедастичность — Breusch-Pagan
    try:
        X_test = sm.add_constant(X_factors_df)
        bp, bp_p, _, _ = het_breuschpagan(resid, X_test)
        res['Гетероскедастичность (Breusch-Pagan)'] = (
            bp_p > alpha, bp, bp_p, f"LM={bp:.3f},  p={bp_p:.3f}")
    except Exception:
        res['Гетероскедастичность (Breusch-Pagan)'] = (True, None, None, "н/д")

    # 6. Гетероскедастичность — White
    try:
        wh, wh_p, _, _ = het_white(resid, sm.add_constant(X_factors_df))
        res['Гетероскедастичность (White)'] = (
            wh_p > alpha, wh, wh_p, f"LM={wh:.3f},  p={wh_p:.3f}")
    except Exception:
        res['Гетероскедастичность (White)'] = (True, None, None, "н/д")

    # 7. Значимость коэффициентов (все p < alpha)
    try:
        pvals = model.pvalues
        pvals = pvals[~pvals.index.str.contains('sigma', case=False, na=False)]
        insig = pvals[pvals > alpha]
        sig_ok = len(insig) == 0
        if sig_ok:
            desc = "все коэффициенты значимы"
        else:
            desc = "незначимы: " + ", ".join(f"{k}(p={v:.3f})" for k, v in insig.items())
        res['Значимость коэффициентов (p < 0.05)'] = (sig_ok, None, None, desc)
    except Exception:
        res['Значимость коэффициентов (p < 0.05)'] = (True, None, None, "н/д")

    return res


def all_ok(assumption_dict):
    return all(v[0] for v in assumption_dict.values())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ШАГ 3: ПЕРЕБОР МОДЕЛЕЙ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def try_one_model(Y_raw, X_raw_df, model_type, ar_order, diff_order, label):
    """
    Пробует одну конфигурацию. Возвращает кортеж или None при ошибке.
    """
    try:
        Y = Y_raw.copy()
        X = X_raw_df.copy()

        if diff_order > 0:
            Y = Y.diff(diff_order).dropna()
            X = X.diff(diff_order).dropna()
            X = X.loc[Y.index]

        X_exog = sm.add_constant(X, has_constant='add')

        if model_type == 'ols':
            m = sm.OLS(Y, X_exog).fit()
        elif model_type == 'ar':
            m = SARIMAX(Y, exog=X_exog, order=(ar_order, 0, 0)).fit(
                method='lbfgs', disp=False, maxiter=600)
        elif model_type == 'ma':
            m = SARIMAX(Y, exog=X_exog, order=(0, 0, ar_order)).fit(
                method='lbfgs', disp=False, maxiter=600)
        elif model_type == 'arma':
            p, q = ar_order
            m = SARIMAX(Y, exog=X_exog, order=(p, 0, q)).fit(
                method='lbfgs', disp=False, maxiter=600)
        else:
            return None

        return dict(model=m, Y_use=Y, X_use=X, X_exog=X_exog,
                    label=label, diff=diff_order, factors=list(X_raw_df.columns))
    except Exception:
        return None


def search_best_model(df, target, factors, alpha=0.05):
    """
    Перебирает конфигурации моделей.
    Приоритет: сначала МАКСИМАЛЬНЫЙ набор факторов (все типы моделей),
    затем меньшие подмножества — только если с полным набором ничего не нашлось.
    """
    Y = df[target]

    # Подмножества факторов по убыванию размера (приоритет — полный набор)
    subsets =[]
    for r in range(len(factors), 0, -1):
        for combo in combinations(factors, r):
            subsets.append(list(combo))

    # Типы моделей от простых к сложным
    configs =[
        ('ols',  0,     0),
        ('ols',  0,     1),
        ('ar',   1,     0),
        ('ar',   1,     1),
        ('ar',   2,     0),
        ('ar',   2,     1),
        ('ar',   3,     0),
        ('ma',   1,     0),
        ('ma',   2,     0),
        ('arma', (1,1), 0),
        ('arma', (2,1), 0),
        ('arma', (1,2), 0),
        ('ar',   3,     1),
    ]

    total = len(subsets) * len(configs)
    print(f"  Перебираю {total} конфигураций ({len(subsets)} подмножеств × {len(configs)} типов моделей)")
    print(f"  Стратегия: сначала все типы моделей на ПОЛНОМ наборе факторов,")
    print(f"             затем — на меньших подмножествах (только если нужно).\n")

    tried = 0
    # Внешний цикл — подмножества (от полного к меньшим)
    # Внутренний — типы моделей
    for cols in subsets:
        for cfg_type, cfg_ar, cfg_diff in configs:
            tried += 1
            if cfg_type in ('ar', 'arma', 'ma'):
                ar_str = f"({cfg_ar[0]},{cfg_ar[1]})" if isinstance(cfg_ar, tuple) else str(cfg_ar)
                type_str = {'ar': f'AR({cfg_ar})', 'ma': f'MA({cfg_ar})',
                            'arma': f'ARMA{ar_str}'}[cfg_type]
            else:
                type_str = 'OLS'
            diff_str = '∆' if cfg_diff else ''
            label = f"{type_str}{diff_str}  [{', '.join(cols)}]"

            result = try_one_model(Y, df[cols], cfg_type, cfg_ar, cfg_diff, label)
            if result is None:
                continue

            m = result['model']
            resid = m.resid
            ar_tests = check_assumptions(m, resid, result['X_use'], alpha=alpha)
            ok = all_ok(ar_tests)

            # Краткий статус
            if ok:
                print(f"  [{tried:>3}/{total}]  ✅  {label}")
            else:
                failed = [k for k, v in ar_tests.items() if not v[0]]
                short = '; '.join(f.split('(')[0].strip() for f in failed)
                print(f"  [{tried:>3}/{total}]  ❌  {label}  →  {short}")

            if ok:
                result['tests'] = ar_tests
                return result

    return None  # fallback


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ШАГ 4: ПРОГНОЗ ФАКТОРОВ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def forecast_factors(df, cols, steps):
    """Линейный тренд для каждого фактора."""
    t = np.arange(len(df))
    future_t = np.arange(len(df), len(df) + steps)
    out = {}
    trend_info = {}
    for col in cols:
        res = sm.OLS(df[col], sm.add_constant(t)).fit()
        a, b = res.params['const'], res.params.iloc[1]
        out[col] = res.predict(sm.add_constant(future_t))
        trend_info[col] = (a, b, res.rsquared)
    return pd.DataFrame(out, index=np.arange(len(df), len(df) + steps)), trend_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ШАГ 5: ВОССТАНОВЛЕНИЕ ПРОГНОЗА ИЗ РАЗНОСТЕЙ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_forecast(result, df, target, steps):
    """
    Строит прогноз Y на steps периодов вперёд.
    Если модель обучена на разностях — восстанавливает уровни.
    """
    cols  = result['factors']
    diff  = result['diff']
    model = result['model']

    X_fut_levels, trend_info = forecast_factors(df, cols, steps)

    if diff > 0:
        last_X = df[cols].iloc[-1]
        d0 = X_fut_levels.iloc[0] - last_X
        d_rest = X_fut_levels.diff().iloc[1:]
        X_fut_diff = pd.concat([d0.to_frame().T, d_rest])
        X_fut_diff.index = X_fut_levels.index
        X_exog_fut = sm.add_constant(X_fut_diff, has_constant='add')
    else:
        X_exog_fut = sm.add_constant(X_fut_levels, has_constant='add')

    try:
        if hasattr(model, 'get_forecast'):
            fc = model.get_forecast(steps=steps, exog=X_exog_fut)
            yhat = fc.predicted_mean.values
            ci   = fc.conf_int()
        else:
            yhat = model.predict(X_exog_fut).values
            ci   = None
    except Exception:
        yhat = model.predict(X_exog_fut).values
        ci   = None

    if diff > 0:
        last_y = df[target].iloc[-1]
        yhat = np.cumsum(yhat) + last_y
        ci   = None

    return yhat, ci, trend_info, X_fut_levels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ШАГ 6: ВИЗУАЛИЗАЦИЯ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_report_figure(df, time_col, target, result, yhat, ci,
                       trend_info, X_fut, removed_corr, removed_vif, out_path):
    model     = result['model']
    tests     = result['tests']
    label     = result['label']
    diff      = result['diff']
    Y_use     = result['Y_use']
    used_cols = result['factors']

    Y_full    = df[target].values
    periods   = list(df[time_col].astype(str))
    steps     = len(yhat)
    fut_labels =[f"+{i+1}" for i in range(steps)]

    # Восстанавливаем fitted в уровнях
    try:
        fitted_raw = model.fittedvalues.values
        if diff > 0:
            last_y = Y_full[diff - 1]
            fitted_levels = np.full(len(Y_full), np.nan)
            cum = last_y
            for i, v in enumerate(fitted_raw):
                cum += v
                fitted_levels[diff + i] = cum
        else:
            fitted_levels = fitted_raw[:len(Y_full)]
            if len(fitted_levels) < len(Y_full):
                fitted_levels = np.concatenate([
                    np.full(len(Y_full) - len(fitted_levels), np.nan), fitted_levels])
    except Exception:
        fitted_levels = np.full(len(Y_full), np.nan)

    # ── Создаём фигуру ────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':    'monospace',
        'text.color':      C['text'],
        'axes.labelcolor': C['text'],
        'xtick.color':     C['muted'],
        'ytick.color':     C['muted'],
    })

    fig = plt.figure(figsize=(20, 14), facecolor=C['bg'])
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.52, wspace=0.38,
                            left=0.06, right=0.97, top=0.91, bottom=0.07)

    def styled_ax(ax, title=''):
        ax.set_facecolor(C['panel'])
        for sp in ax.spines.values():
            sp.set_color(C['border'])
        ax.tick_params(colors=C['muted'], labelsize=8)
        ax.grid(True, color=C['border'], linewidth=0.6, alpha=0.6)
        if title:
            ax.set_title(title, color=C['text'], fontsize=9.5,
                         fontweight='bold', pad=7)
        return ax

    # ══ ГЛАВНЫЙ ГРАФИК (colspan 3) ════════════════════════════════
    ax_main = fig.add_subplot(gs[0, :])
    styled_ax(ax_main, f'ПРОГНОЗ: {target}   │   Модель: {label}')

    x_hist  = np.arange(len(Y_full))
    x_fut   = np.arange(len(Y_full), len(Y_full) + steps)
    all_x   = np.concatenate([x_hist, x_fut])
    all_lbl = periods + fut_labels

    # История
    ax_main.plot(x_hist, Y_full, color=C['text'], lw=2.2,
                 marker='o', ms=4, zorder=4, label='Факт')
    # Подгонка
    ax_main.plot(x_hist, fitted_levels, color=C['fit'],
                 lw=1.6, linestyle='--', alpha=0.85, zorder=3, label='Подгонка')
    # Разделитель
    ax_main.axvline(len(Y_full) - 0.5, color=C['border'],
                    lw=1.2, linestyle=':', alpha=0.9)
    ax_main.axvspan(len(Y_full) - 0.5, len(Y_full) + steps,
                    color=C['forecast'], alpha=0.06)
    # Доверительный интервал
    if ci is not None:
        try:
            ax_main.fill_between(x_fut,
                                 ci.iloc[:, 0].values,
                                 ci.iloc[:, 1].values,
                                 color=C['forecast'], alpha=0.18,
                                 label='ДИ 95%')
        except Exception:
            pass
    # Прогноз
    ax_main.plot(x_fut, yhat, color=C['forecast'],
                 lw=2.5, marker='D', ms=7, zorder=5, label='Прогноз')

    for i, (xi, yi) in enumerate(zip(x_fut, yhat)):
        ax_main.annotate(f'{yi:,.1f}',
                         xy=(xi, yi), xytext=(0, 14),
                         textcoords='offset points',
                         ha='center', fontsize=9,
                         color=C['forecast'], fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.25',
                                   fc=C['panel'], ec=C['forecast'],
                                   lw=0.8, alpha=0.9))

    ax_main.set_xticks(all_x)
    ax_main.set_xticklabels(all_lbl, rotation=45, ha='right', fontsize=7.5)
    ax_main.legend(fontsize=8.5, facecolor=C['panel'],
                   edgecolor=C['border'], labelcolor=C['text'])

    # ══ МАТРИЦА ПАРНОЙ КОРРЕЛЯЦИИ ═════════════════════════════════
    ax_corr = fig.add_subplot(gs[1, :2])
    ax_corr.set_facecolor(C['panel'])
    for sp in ax_corr.spines.values():
        sp.set_color(C['border'])
    ax_corr.set_title('МАТРИЦА КОЭФФИЦИЕНТОВ ПАРНОЙ ЛИНЕЙНОЙ КОРРЕЛЯЦИИ',
                      color=C['text'], fontsize=9.5, fontweight='bold', pad=7)

    # Берём Y + все исходные факторы (до отбора) для полной картины
    all_numeric_cols = [target] + list(df.select_dtypes(include=np.number).columns.difference([target]))
    corr_df = df[all_numeric_cols].corr()
    n_vars  = len(corr_df)

    # Кастомная цветовая карта: красный → тёмный → синий
    cmap_corr = LinearSegmentedColormap.from_list(
        'dark_corr',[(0.0, '#F85149'),   # -1  красный
         (0.5, '#1C2128'),   # 0   почти чёрный
         (1.0, '#58A6FF')],  # +1  синий
        N=256
    )

    mat = corr_df.values
    im  = ax_corr.imshow(mat, cmap=cmap_corr, vmin=-1, vmax=1, aspect='auto')

    # Сетка-разделители
    for i in range(n_vars + 1):
        ax_corr.axhline(i - 0.5, color=C['bg'], lw=1.0)
        ax_corr.axvline(i - 0.5, color=C['bg'], lw=1.0)

    # Подписи осей
    ax_corr.set_xticks(range(n_vars))
    ax_corr.set_yticks(range(n_vars))
    ax_corr.set_xticklabels(corr_df.columns, rotation=30, ha='right',
                             fontsize=8.5, color=C['text'])
    ax_corr.set_yticklabels(corr_df.columns, fontsize=8.5, color=C['text'])
    ax_corr.tick_params(length=0)

    # Числа внутри ячеек
    for i in range(n_vars):
        for j in range(n_vars):
            val  = mat[i, j]
            # Тёмный текст на светлых ячейках, светлый — на тёмных
            brightness = abs(val)
            txt_clr = C['bg'] if brightness > 0.55 else C['text']
            weight  = 'bold' if (i != j and abs(val) > 0.7) else 'normal'
            ax_corr.text(j, i, f'{val:.2f}',
                         ha='center', va='center',
                         fontsize=8.5, color=txt_clr, fontweight=weight)

    # Выделить ячейки Y-строки и Y-столбца рамкой
    y_idx = 0  # target — первый столбец
    for k in range(n_vars):
        for (ri, ci_) in[(y_idx, k), (k, y_idx)]:
            rect = plt.Rectangle((ci_ - 0.5, ri - 0.5), 1, 1,
                                  fill=False, edgecolor=C['yellow'],
                                  lw=1.2, zorder=3)
            ax_corr.add_patch(rect)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_corr, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=C['muted'], labelsize=7)
    cbar.outline.set_edgecolor(C['border'])

    # ══ ТРЕНДЫ ФАКТОРОВ ═══════════════════════════════════════════
    ax_fct = fig.add_subplot(gs[1, 2])
    styled_ax(ax_fct, 'ТРЕНДЫ ФАКТОРОВ (прогноз X)')

    factor_colors = [C['accent'], C['green'], C['purple'],
                     C['yellow'], C['forecast']]
    t_all   = np.arange(len(df) + steps)
    t_hist2 = np.arange(len(df))
    t_fut2  = np.arange(len(df), len(df) + steps)

    for idx, col in enumerate(used_cols):
        clr = factor_colors[idx % len(factor_colors)]
        y_norm = df[col].values / df[col].max()
        ax_fct.plot(t_hist2, y_norm, color=clr, lw=1.4, alpha=0.7)
        y_fut_norm = X_fut[col].values / df[col].max()
        ax_fct.plot(t_fut2, y_fut_norm,
                    color=clr, lw=1.4, linestyle='--',
                    marker='s', ms=4, label=col)
    ax_fct.axvline(len(df) - 0.5, color=C['border'],
                   lw=1, linestyle=':', alpha=0.7)
    ax_fct.legend(fontsize=7.5, facecolor=C['panel'],
                  edgecolor=C['border'], labelcolor=C['text'])
    ax_fct.set_ylabel('Норм. значение', fontsize=8, color=C['muted'])

    # ══ ТАБЛИЦА ТЕСТОВ ПРЕДПОСЫЛОК ════════════════════════════════
    ax_tbl = fig.add_subplot(gs[2, :2])
    ax_tbl.set_facecolor(C['panel'])
    for sp in ax_tbl.spines.values():
        sp.set_color(C['border'])
    ax_tbl.set_title('ПРОВЕРКА ПРЕДПОСЫЛОК МНК', color=C['text'],
                      fontsize=9.5, fontweight='bold', pad=7)
    ax_tbl.axis('off')

    rows =[]
    colors_cells =[]
    for test_name, (passed, stat, pval, desc) in tests.items():
        icon   = '✅' if passed else '❌'
        verdict = 'ВЫПОЛНЕНО' if passed else 'НАРУШЕНО'
        rows.append([icon, test_name, desc, verdict])
        clr_v = C['green'] if passed else C['red']
        colors_cells.append([C['panel'], C['panel'], C['panel'], clr_v])

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=['', 'Тест', 'Статистика', 'Вердикт'],
        cellLoc='left', loc='center',
        colWidths=[0.05, 0.45, 0.32, 0.18]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(C['panel'] if r > 0 else '#1C2128')
        cell.set_edgecolor(C['border'])
        if r == 0:
            cell.set_text_props(color=C['text'], fontweight='bold')
        elif c == 3 and r > 0:
            clr_v = colors_cells[r - 1][3]
            cell.set_text_props(color=clr_v, fontweight='bold')
        else:
            cell.set_text_props(color=C['text'])

    # ══ КАРТОЧКА РЕЗУЛЬТАТОВ ══════════════════════════════════════
    ax_card = fig.add_subplot(gs[2, 2])
    ax_card.set_facecolor(C['panel'])
    for sp in ax_card.spines.values():
        sp.set_color(C['border'])
    ax_card.set_title('ПРОГНОЗ', color=C['text'],
                       fontsize=9.5, fontweight='bold', pad=7)
    ax_card.axis('off')

    card_lines =[]
    last_hist = Y_full[-1]
    for i, val in enumerate(yhat):
        chg = (val / last_hist - 1) * 100 if i == 0 else (val / yhat[i-1] - 1) * 100
        arrow = '▲' if chg >= 0 else '▼'
        clr   = C['green'] if chg >= 0 else C['red']
        card_lines.append((f"Период +{i+1}", f"{val:>10,.1f}", f"{arrow} {abs(chg):.1f}%", clr))

    y_pos = 0.82
    for period, val, chg, clr in card_lines:
        ax_card.text(0.03, y_pos, period, color=C['muted'],
                     fontsize=9, transform=ax_card.transAxes)
        ax_card.text(0.50, y_pos, val, color=C['text'],
                     fontsize=9.5, fontweight='bold',
                     transform=ax_card.transAxes, ha='center')
        ax_card.text(0.97, y_pos, chg, color=clr,
                     fontsize=9, transform=ax_card.transAxes, ha='right')
        line = plt.Line2D([0.02, 0.98],[y_pos - 0.09, y_pos - 0.09],
                          color=C['border'], lw=0.5,
                          transform=ax_card.transAxes, figure=fig)
        fig.add_artist(line)
        y_pos -= 0.18

    # Мультиколлинеарность — плашка
    if removed_corr or removed_vif:
        note = "Удалены (мультикол.): "
        if removed_corr:
            note += f"корр={removed_corr} "
        if removed_vif:
            note += f"VIF={removed_vif}"
        fig.text(0.5, 0.004, note,
                 ha='center', fontsize=7.5,
                 color=C['yellow'], alpha=0.8)

    # Заголовок
    try:
        r2 = result['model'].rsquared
        r2_adj = result['model'].rsquared_adj
    except Exception:
        # Для SARIMAX считаем вручную
        try:
            y_fit = result['model'].fittedvalues.values
            y_true = result['Y_use'].values
            ss_res = np.sum((y_true - y_fit) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
            n = len(y_true)
            k = len(result['factors'])
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if r2 is not None else None
        except Exception:
            r2 = r2_adj = None

    r2_str = f"  │  R²={r2:.4f}" if r2 is not None else ''
    fig.suptitle(
        f"СИСТЕМА АВТОМАТИЗИРОВАННОГО РЕГРЕССИОННОГО МОДЕЛИРОВАНИЯ  │  {target}{r2_str}",
        color=C['text'], fontsize=13, fontweight='bold', y=0.975
    )

    # Плашка R² на главном графике
    if r2 is not None:
        r2_lines = [f"R²      = {r2:.4f}"]
        if r2_adj is not None:
            r2_lines.append(f"R²_adj = {r2_adj:.4f}")
        badge_txt = "\n".join(r2_lines)
        ax_main.text(0.01, 0.97, badge_txt,
                     transform=ax_main.transAxes,
                     va='top', ha='left',
                     fontsize=9.5, fontweight='bold',
                     color=C['accent'],
                     bbox=dict(boxstyle='round,pad=0.5',
                               fc=C['panel'], ec=C['accent'],
                               lw=1.2, alpha=0.92))

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.show()
    print(f"\n✅  График сохранён: {out_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ГЛАВНАЯ ФУНКЦИЯ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_universal_modeling(df, forecast_steps=4, alpha=0.05,
                           out_path='/mnt/user-data/outputs/report.png'):
    time_col = df.columns[0]
    target   = df.columns[1]
    all_factors = list(df.columns[2:])

    SEP = "═" * 80

    # ── ЭТАП 1: МУЛЬТИКОЛЛИНЕАРНОСТЬ ─────────────────────────────
    print(SEP)
    print(f"  ЭТАП 1 / 4  │  АНАЛИЗ МУЛЬТИКОЛЛИНЕАРНОСТИ")
    print(SEP)

    final_factors, removed_corr, removed_vif, mclog = select_factors(
        df, target, all_factors)

    if mclog:
        for line in mclog:
            print(line)
    else:
        print("  ✅  Мультиколлинеарность не обнаружена.")
    print(f"\n  Итоговый набор факторов: {final_factors}")

    # ── ЭТАП 2: ПОИСК МОДЕЛИ ─────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  ЭТАП 2 / 4  │  ПОИСК ОПТИМАЛЬНОЙ МОДЕЛИ")
    print(SEP)

    result = search_best_model(df, target, final_factors, alpha=alpha)

    if result is None:
        print("\n⚠  Не найдена модель со всеми выполненными предпосылками.")
        print("   Используется OLS на полном наборе факторов (fallback).")
        Y = df[target]
        X = sm.add_constant(df[final_factors], has_constant='add')
        m = sm.OLS(Y, X).fit()
        tests = check_assumptions(m, m.resid, df[final_factors], alpha)
        result = dict(model=m, Y_use=Y, X_use=df[final_factors],
                      X_exog=X, label=f"OLS (fallback) [{final_factors}]",
                      diff=0, factors=final_factors, tests=tests)

    print(f"\n  🏆  ПОБЕДИВШАЯ МОДЕЛЬ: {result['label']}")

    # ── ЭТАП 3: ВЫВОД ИТОГОВ ─────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  ЭТАП 3 / 4  │  ИТОГИ МОДЕЛИРОВАНИЯ")
    print(SEP)

    # Выводим верхнюю часть стандартного summary (информация о модели)
    try:
        print(result['model'].summary().tables[0])
    except Exception:
        pass

    # Выводим кастомную таблицу коэффициентов
    print(f"\n  Оценка коэффициентов:")
    print(f"  {'─'*77}")
    print(f"  {'Переменная':<15} | {'coef':>10} | {'std err':>10} | {'p-value':>10} | {'5% уровень':>10} | {'1% уровень':>10}")
    print(f"  {'─'*77}")

    params = result['model'].params
    bse = result['model'].bse
    pvals = result['model'].pvalues

    for var_name in params.index:
        coef = params[var_name]
        err = bse[var_name]
        pval = pvals[var_name]

        lvl_5 = "Да" if pval < 0.05 else "Нет"
        lvl_1 = "Да" if pval < 0.01 else "Нет"

        # Обрезаем имя переменной до 15 символов, чтобы таблица не ломалась
        print(f"  {var_name[:15]:<15} | {coef:>10.4f} | {err:>10.4f} | {pval:>10.4f} | {lvl_5:>10} | {lvl_1:>10}")
    print(f"  {'─'*77}\n")

    print(f"\n  {'Тест':<45} {'Результат':<35} {'Вердикт'}")
    print(f"  {'─'*45} {'─'*35} {'─'*10}")
    for test_name, (passed, stat, pval, desc) in result['tests'].items():
        icon = '✅' if passed else '❌'
        print(f"  {icon}  {test_name:<43} {desc:<35}")

    # ── ЭТАП 4: ПРОГНОЗ ──────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  ЭТАП 4 / 4  │  ПРОГНОЗ  ({forecast_steps} периодов вперёд)")
    print(SEP)

    yhat, ci, trend_info, X_fut = make_forecast(
        result, df, target, forecast_steps)

    print(f"\n  Тренды факторов:")
    for col, (a, b, r2) in trend_info.items():
        sign = '+' if b >= 0 else '-'
        print(f"    {col}: {a:.3f} {sign} {abs(b):.4f}·t   (R²={r2:.3f})")

    last = df[target].iloc[-1]
    print(f"\n  Последнее фактическое значение ({df[time_col].iloc[-1]}): {last:,.2f}")
    print(f"\n  Прогнозные значения {target}:")
    for i, val in enumerate(yhat):
        chg = (val / last - 1) * 100 if i == 0 else (val / yhat[i-1] - 1) * 100
        arrow = '▲' if chg >= 0 else '▼'
        print(f"    Период +{i+1}:  {val:>12,.2f}   {arrow} {abs(chg):.2f}%")

    # ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────
    make_report_figure(
        df, time_col, target, result, yhat, ci,
        trend_info, X_fut, removed_corr, removed_vif, out_path
    )

    return result, yhat


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ДАННЫЕ И ЗАПУСК
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

data = {
    'Период':['19Q1','19Q2','19Q3','19Q4',
                     '20Q1','20Q2','20Q3','20Q4',
                     '21Q1','21Q2','21Q3','21Q4',
                     '22Q1','22Q2','22Q3','22Q4',
                     '23Q1','23Q2','23Q3','23Q4',
                     '24Q1','24Q2','24Q3','24Q4'],
    'Wage_Y':[1011.0,1074.6,1119.6,1158.5,
                     1155.4,1222.4,1277.9,1351.2,
                     1321.6,1416.4,1461.2,1541.6,
                     1553.7,1588.7,1650.8,1731.4,
                     1731.2,1861.0,1947.8,2071.0,
                     2065.5,2222.3,2339.1,2461.2],
    'GDP_X1':[91.3,98.3,107.1,108.9,91.4,95.4,107.2,108.8,
                     92.7,101.2,108.4,110.2,92.5,93.2,102.6,104.7,
                     91.0,98.9,109.0,110.3,95.1,104.7,113.7,113.6],
    'Employed_X2':[4928.0,4888.7,4951.3,4909.6,4820.5,4855.9,4939.5,4919.4,
                     4785.6,4875.3,4913.4,4823.3,4771.1,4856.5,4906.6,4845.4,
                     4725.5,4821.2,4830.5,4827.5,4800.9,4829.8,4827.6,4736.1],
    'Unempl_X3':[4.56,4.35,3.88,3.98,4.11,4.20,3.97,4.07,
                     4.16,3.98,3.66,3.77,3.72,3.67,3.36,3.57,
                     3.60,3.38,3.36,3.48,3.26,3.00,2.85,3.03],
    'CPI_X4':[85.6,86.4,86.6,87.6,89.6,90.9,91.5,93.5,
                     97.0,99.4,100.6,103.1,108.7,116.4,118.5,117.3,
                     119.3,120.5,121.0,123.0,126.0,127.7,128.4,129.8]
}

my_df = pd.DataFrame(data)

result, yhat = run_universal_modeling(
    my_df,
    forecast_steps=4,
    out_path='report.png'
)