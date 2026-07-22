"""
retail_dashboard.py
====================
Retail Chain Executive Dashboard (advanced level) — single-file build
-----------------------------------------------------------------------
A multi-page, cross-filtering, drill-down BI dashboard built with
PyQt5 + matplotlib, styled to resemble Power BI's report UX:

  * Left navigation rail -> switches between report pages
  * Global cross-filter state (Region -> Store -> Product hierarchy
    + Category) shared across every page and shown in a breadcrumb bar
  * Clicking any chart bar drills down / cross-filters every other
    visual on every page (like Power BI's "Apply as filter" behavior)
  * KPI cards with YoY growth and Target-vs-Actual variance
  * A semicircular gauge visual for target achievement

This file merges what were previously four modules (data_model.py,
widgets.py, pages.py, main.py) into one self-contained script, organized
into clearly labeled sections in dependency order:

  1. DATA MODEL       - synthetic data generation + aggregation helpers
  2. UI WIDGETS        - palette, KPI cards, chart canvas, breadcrumb
  3. REPORT PAGES      - the four dashboard pages
  4. MAIN WINDOW / APP - navigation shell, cross-filter state, entry point

Run:
    python3 retail_dashboard.py

Requirements:
    pip install PyQt5 matplotlib numpy pandas
"""

import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFrame, QComboBox,
    QGridLayout, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"


# ======================================================================
# 1. DATA MODEL
# ======================================================================
# Synthetic data generation + aggregation helpers for the Retail Chain
# Executive Dashboard. Simulates a realistic star-schema-like structure
# (Region -> Store -> Product -> monthly Sales/Target facts) across two
# fiscal years so Year-over-Year (YoY) comparisons are meaningful.
# ======================================================================

RNG = np.random.default_rng(42)

REGIONS = {
    "North":  ["North-01 Lahore", "North-02 Islamabad", "North-03 Faisalabad"],
    "South":  ["South-01 Karachi", "South-02 Hyderabad", "South-03 Sukkur"],
    "East":   ["East-01 Multan", "East-02 Bahawalpur"],
    "West":   ["West-01 Quetta", "West-02 Peshawar", "West-03 Abbottabad"],
}

PRODUCTS = {
    "Electronics": ["Smartphones", "Laptops", "Accessories"],
    "Apparel":     ["Menswear", "Womenswear", "Footwear"],
    "Grocery":     ["Packaged Foods", "Beverages"],
    "Home":        ["Furniture", "Kitchenware"],
}

MONTHS = pd.date_range("2024-01-01", "2025-12-01", freq="MS")

_BASE_STORE_INDEX = {
    store: i for i, store in enumerate(
        s for stores in REGIONS.values() for s in stores
    )
}
_BASE_PRODUCT_INDEX = {
    prod: i for i, prod in enumerate(
        p for prods in PRODUCTS.values() for p in prods
    )
}


def _seasonal_factor(month: int) -> float:
    """Retail seasonality: bump in Nov/Dec, dip in Feb."""
    bumps = {11: 1.35, 12: 1.55, 1: 0.85, 2: 0.8}
    return bumps.get(month, 1.0)


def generate_fact_table() -> pd.DataFrame:
    """Builds the full fact table: one row per Region/Store/Category/Product/Month."""
    rows = []
    for region, stores in REGIONS.items():
        region_strength = RNG.uniform(0.85, 1.25)
        for store in stores:
            store_idx = _BASE_STORE_INDEX[store]
            store_strength = RNG.uniform(0.8, 1.3)
            for category, prods in PRODUCTS.items():
                for product in prods:
                    prod_idx = _BASE_PRODUCT_INDEX[product]
                    base = 45_000 * region_strength * store_strength
                    base *= RNG.uniform(0.6, 1.4)  # product popularity variance
                    growth_rate = RNG.uniform(0.06, 0.22)  # YoY growth trend

                    for date in MONTHS:
                        year = date.year
                        month = date.month
                        yr_offset = year - 2024
                        trend = (1 + growth_rate) ** yr_offset
                        seasonal = _seasonal_factor(month)
                        noise = RNG.normal(1.0, 0.08)

                        sales = max(0, base * trend * seasonal * noise)
                        # Target set with slight optimism vs prior-year actual pace
                        target = base * trend * seasonal * RNG.uniform(0.95, 1.1)

                        rows.append((
                            region, store, category, product,
                            date, year, month, round(sales, 2), round(target, 2)
                        ))

    df = pd.DataFrame(rows, columns=[
        "Region", "Store", "Category", "Product", "Date",
        "Year", "Month", "Sales", "Target"
    ])
    return df


class DataModel:
    """Wraps the fact table and exposes filtered aggregation helpers
    used throughout the dashboard for cross-filtering + drill-down."""

    def __init__(self):
        self.df = generate_fact_table()
        self.current_year = int(self.df["Year"].max())
        self.prior_year = self.current_year - 1

    # ---------- filtering ----------
    def filtered(self, region=None, store=None, category=None, product=None, year=None):
        df = self.df
        if region:
            df = df[df["Region"] == region]
        if store:
            df = df[df["Store"] == store]
        if category:
            df = df[df["Category"] == category]
        if product:
            df = df[df["Product"] == product]
        if year:
            df = df[df["Year"] == year]
        return df

    # ---------- KPI helpers ----------
    def kpis(self, region=None, store=None, category=None, product=None):
        cur = self.filtered(region, store, category, product, year=self.current_year)
        prior = self.filtered(region, store, category, product, year=self.prior_year)

        total_sales = cur["Sales"].sum()
        total_target = cur["Target"].sum()
        prior_sales = prior["Sales"].sum()

        achievement = (total_sales / total_target * 100) if total_target else 0
        yoy = ((total_sales - prior_sales) / prior_sales * 100) if prior_sales else 0
        variance = total_sales - total_target

        return {
            "sales": total_sales,
            "target": total_target,
            "variance": variance,
            "achievement": achievement,
            "yoy": yoy,
            "prior_sales": prior_sales,
        }

    # ---------- grouped aggregations for charts ----------
    def by_region(self, year=None):
        year = year or self.current_year
        df = self.filtered(year=year)
        return df.groupby("Region")[["Sales", "Target"]].sum().reindex(REGIONS.keys())

    def by_store(self, region, year=None):
        year = year or self.current_year
        df = self.filtered(region=region, year=year)
        stores = REGIONS[region]
        return df.groupby("Store")[["Sales", "Target"]].sum().reindex(stores)

    def by_product(self, region=None, store=None, year=None):
        year = year or self.current_year
        df = self.filtered(region=region, store=store, year=year)
        return df.groupby("Product")[["Sales", "Target"]].sum().sort_values(
            "Sales", ascending=False
        )

    def by_category(self, region=None, store=None, year=None):
        year = year or self.current_year
        df = self.filtered(region=region, store=store, year=year)
        return df.groupby("Category")[["Sales", "Target"]].sum().sort_values(
            "Sales", ascending=False
        )

    def monthly_trend(self, region=None, store=None, category=None, product=None):
        """Returns two 12-length series (current year, prior year) indexed by month 1..12."""
        cur = self.filtered(region, store, category, product, year=self.current_year)
        prior = self.filtered(region, store, category, product, year=self.prior_year)
        cur_m = cur.groupby("Month")["Sales"].sum().reindex(range(1, 13), fill_value=0)
        prior_m = prior.groupby("Month")["Sales"].sum().reindex(range(1, 13), fill_value=0)
        return cur_m, prior_m

    def store_list(self, region):
        return REGIONS.get(region, [])

    def product_list(self, region=None, store=None):
        df = self.filtered(region=region, store=store)
        return sorted(df["Product"].unique().tolist())


# ======================================================================
# 2. UI WIDGETS
# ======================================================================
# Reusable, Power-BI-styled UI building blocks: color palette, KPI cards,
# a clickable matplotlib canvas (for cross-filter drill-down), and a
# breadcrumb filter bar.
# ======================================================================

class Palette:
    bg = "#F3F2F1"
    card = "#FFFFFF"
    text = "#252423"
    subtext = "#605E5C"
    accent = "#118DFF"        # Power BI signature blue
    accent_dark = "#0F6CBD"
    good = "#107C10"
    bad = "#D13438"
    warn = "#FFB900"
    grid = "#E1DFDD"
    sidebar = "#20242B"
    sidebar_active = "#118DFF"
    series = ["#118DFF", "#12239E", "#E66C37", "#6B007B", "#E044A7",
              "#744EC2", "#D9B300", "#D64550"]


def shadow(blur=18, dx=0, dy=2, alpha=45):
    eff = QGraphicsDropShadowEffect()
    eff.setBlurRadius(blur)
    eff.setOffset(dx, dy)
    eff.setColor(QColor(0, 0, 0, alpha))
    return eff


# ----------------------------------------------------------------------
# KPI Card
# ----------------------------------------------------------------------
class KpiCard(QFrame):
    """A single Power BI style KPI tile with a title, big value, and a
    colored delta indicator (e.g. YoY % or Achievement %)."""

    def __init__(self, title, value="--", delta_text=None, delta_positive=True, parent=None):
        super().__init__(parent)
        self.setObjectName("KpiCard")
        self.setStyleSheet(f"""
            #KpiCard {{
                background-color: {Palette.card};
                border-radius: 10px;
                border: 1px solid {Palette.grid};
            }}
        """)
        self.setGraphicsEffect(shadow())
        self.setMinimumHeight(96)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(4)

        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet(f"color:{Palette.subtext}; letter-spacing:0.5px;")
        self.title_lbl.setFont(QFont("Segoe UI", 9, QFont.DemiBold))

        self.value_lbl = QLabel(value)
        self.value_lbl.setStyleSheet(f"color:{Palette.text};")
        self.value_lbl.setFont(QFont("Segoe UI", 20, QFont.Bold))

        self.delta_lbl = QLabel(delta_text or "")
        self.delta_lbl.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
        self._set_delta_color(delta_positive)

        lay.addWidget(self.title_lbl)
        lay.addWidget(self.value_lbl)
        lay.addWidget(self.delta_lbl)

    def _set_delta_color(self, positive):
        color = Palette.good if positive else Palette.bad
        self.delta_lbl.setStyleSheet(f"color:{color};")

    def update_values(self, value, delta_text=None, delta_positive=True):
        self.value_lbl.setText(value)
        if delta_text is not None:
            self.delta_lbl.setText(delta_text)
            self._set_delta_color(delta_positive)


# ----------------------------------------------------------------------
# Matplotlib canvas with click-to-drill support
# ----------------------------------------------------------------------
class ChartCanvas(FigureCanvas):
    """Matplotlib canvas emitting `bar_clicked(label)` when a bar/point
    with a category label is clicked -> powers cross-filtering."""

    bar_clicked = pyqtSignal(str)

    def __init__(self, width=5, height=3.2, dpi=100, parent=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=Palette.card)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self._label_map = {}  # matplotlib artist -> category label
        self.mpl_connect("pick_event", self._on_pick)
        self.setStyleSheet("background-color:transparent;")

    def _on_pick(self, event):
        label = self._label_map.get(event.artist)
        if label:
            self.bar_clicked.emit(label)

    def register_pickable(self, artist, label):
        artist.set_picker(True)
        self._label_map[artist] = label

    def clear(self):
        self.ax.clear()
        self._label_map = {}

    def style_axes(self, title=None, ylabel=None):
        self.ax.set_facecolor(Palette.card)
        for spine in ("top", "right"):
            self.ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            self.ax.spines[spine].set_color(Palette.grid)
        self.ax.tick_params(colors=Palette.subtext, labelsize=8)
        self.ax.grid(axis="y", color=Palette.grid, linewidth=0.8, zorder=0)
        self.ax.set_axisbelow(True)
        if title:
            self.ax.set_title(title, color=Palette.text, fontsize=11,
                               fontweight="bold", loc="left", pad=10)
        if ylabel:
            self.ax.set_ylabel(ylabel, color=Palette.subtext, fontsize=8)


class ChartCard(QFrame):
    """A white rounded card wrapping a chart canvas + optional header, matching
    the visual language of a Power BI report visual container."""

    def __init__(self, title="", height=320, parent=None):
        super().__init__(parent)
        self.setObjectName("ChartCard")
        self.setStyleSheet(f"""
            #ChartCard {{
                background-color: {Palette.card};
                border-radius: 10px;
                border: 1px solid {Palette.grid};
            }}
        """)
        self.setGraphicsEffect(shadow())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)

        if title:
            header = QLabel(title)
            header.setFont(QFont("Segoe UI", 10, QFont.Bold))
            header.setStyleSheet(f"color:{Palette.text}; border:none;")
            lay.addWidget(header)

        self.canvas = ChartCanvas(height=height / 100)
        lay.addWidget(self.canvas)
        self.setMinimumHeight(height)


# ----------------------------------------------------------------------
# Breadcrumb filter bar (shows current drill path, click to reset level)
# ----------------------------------------------------------------------
class Breadcrumb(QFrame):
    segment_clicked = pyqtSignal(str)  # emits level name clicked: 'all'|'region'|'store'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(6)
        self.lay.addStretch()
        self._buttons = []

    def _make_crumb(self, text, level, active=False):
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        color = Palette.accent if active else Palette.subtext
        weight = "bold" if active else "normal"
        btn.setStyleSheet(f"""
            QPushButton {{
                border: none; background: transparent; color: {color};
                font-weight: {weight}; font-size: 12px; padding: 2px 4px;
            }}
            QPushButton:hover {{ text-decoration: underline; }}
        """)
        btn.clicked.connect(lambda: self.segment_clicked.emit(level))
        return btn

    def set_path(self, region=None, store=None, product=None):
        # clear
        while self.lay.count():
            item = self.lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        crumbs = [("All Regions", "all")]
        if region:
            crumbs.append((region, "region"))
        if store:
            crumbs.append((store, "store"))
        if product:
            crumbs.append((product, "product"))

        for i, (text, level) in enumerate(crumbs):
            is_last = (i == len(crumbs) - 1)
            self.lay.addWidget(self._make_crumb(text, level, active=is_last))
            if not is_last:
                sep = QLabel("›")
                sep.setStyleSheet(f"color:{Palette.subtext};")
                self.lay.addWidget(sep)
        self.lay.addStretch()


# ======================================================================
# 3. REPORT PAGES
# ======================================================================
# The four report pages of the dashboard. Each page exposes:
#   - build(): constructs the static layout (cards, canvases)
#   - refresh(state): redraws all charts/KPIs given the current
#     cross-filter state (region/store/category/product selections)
#
# Every chart bar is pickable; clicking it calls back into the
# DashboardWindow's drill-down handler via Qt signals, which is how
# cross-filtering + hierarchical drill-down (Region -> Store -> Product)
# is implemented, mirroring Power BI's click-to-filter/drill behavior.
# ======================================================================

def _fmt_money(v):
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:,.2f}M"
    if abs(v) >= 1_000:
        return f"${v/1_000:,.1f}K"
    return f"${v:,.0f}"


def _fmt_pct(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:,.1f}%"


MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class BasePage(QWidget):
    """Common scaffolding: emits `drill_requested(level, value)` when a
    chart element is clicked, so the main window can update the shared
    filter state and re-render every page."""

    drill_requested = pyqtSignal(str, str)  # level ('region'|'store'|'product'), value

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.setStyleSheet(f"background-color:{Palette.bg};")

    def section_title(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Segoe UI", 13, QFont.Bold))
        lbl.setStyleSheet(f"color:{Palette.text};")
        return lbl


# ----------------------------------------------------------------------
# 3.1 OVERVIEW PAGE
# ----------------------------------------------------------------------
class OverviewPage(BasePage):
    """Executive summary: KPI tiles, Sales-by-Region (drill entry point),
    12-month current-vs-prior-year trend, and Target-vs-Actual by region."""

    def build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        root.addWidget(self.section_title("Executive Overview"))

        # KPI row
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(14)
        self.kpi_sales = KpiCard("Total Sales (YTD)")
        self.kpi_target = KpiCard("Target (YTD)")
        self.kpi_achv = KpiCard("Achievement")
        self.kpi_yoy = KpiCard("YoY Growth")
        for c in (self.kpi_sales, self.kpi_target, self.kpi_achv, self.kpi_yoy):
            kpi_row.addWidget(c)
        root.addLayout(kpi_row)

        # Chart grid
        grid = QGridLayout()
        grid.setSpacing(14)
        self.region_card = ChartCard("Sales by Region  (click a bar to drill down)")
        self.trend_card = ChartCard("Monthly Trend — Current vs Prior Year")
        self.tva_card = ChartCard("Target vs Actual by Region")
        self.cat_card = ChartCard("Sales by Category")

        grid.addWidget(self.region_card, 0, 0)
        grid.addWidget(self.trend_card, 0, 1)
        grid.addWidget(self.tva_card, 1, 0)
        grid.addWidget(self.cat_card, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        root.addLayout(grid, stretch=1)

        self.region_card.canvas.bar_clicked.connect(
            lambda label: self.drill_requested.emit("region", label))
        self.cat_card.canvas.bar_clicked.connect(
            lambda label: self.drill_requested.emit("category", label))

    def refresh(self, state):
        region, store, category, product = state.region, state.store, state.category, state.product
        k = self.model.kpis(region, store, category, product)

        self.kpi_sales.update_values(_fmt_money(k["sales"]))
        self.kpi_target.update_values(_fmt_money(k["target"]))
        self.kpi_achv.update_values(f"{k['achievement']:.1f}%",
                                     _fmt_pct(k['achievement'] - 100) + " vs target",
                                     k['achievement'] >= 100)
        self.kpi_yoy.update_values(_fmt_pct(k["yoy"]),
                                    f"prior year {_fmt_money(k['prior_sales'])}",
                                    k["yoy"] >= 0)

        self._draw_region_chart(state)
        self._draw_trend_chart(state)
        self._draw_target_vs_actual(state)
        self._draw_category_chart(state)

    def _draw_region_chart(self, state):
        c = self.region_card.canvas
        c.clear()
        df = self.model.by_region()
        x = np.arange(len(df))
        bars = c.ax.bar(x, df["Sales"], color=Palette.accent, width=0.55, zorder=3)
        for i, (region, bar) in enumerate(zip(df.index, bars)):
            if region == state.region:
                bar.set_color(Palette.accent_dark)
            c.register_pickable(bar, region)
        c.ax.set_xticks(x)
        c.ax.set_xticklabels(df.index, fontsize=8)
        c.style_axes(ylabel="Sales")
        c.fig.tight_layout()
        c.draw()

    def _draw_trend_chart(self, state):
        c = self.trend_card.canvas
        c.clear()
        cur, prior = self.model.monthly_trend(state.region, state.store, state.category, state.product)
        x = np.arange(12)
        c.ax.plot(x, prior.values, color=Palette.grid, linewidth=2.2,
                  marker="o", markersize=3, label=f"{self.model.prior_year}")
        c.ax.plot(x, cur.values, color=Palette.accent, linewidth=2.6,
                  marker="o", markersize=3.5, label=f"{self.model.current_year}")
        c.ax.set_xticks(x)
        c.ax.set_xticklabels(MONTH_LABELS, fontsize=7.5)
        c.style_axes(ylabel="Sales")
        c.ax.legend(frameon=False, fontsize=8, loc="upper left")
        c.fig.tight_layout()
        c.draw()

    def _draw_target_vs_actual(self, state):
        c = self.tva_card.canvas
        c.clear()
        df = self.model.by_region()
        x = np.arange(len(df))
        w = 0.35
        b1 = c.ax.bar(x - w/2, df["Target"], width=w, color=Palette.grid, label="Target", zorder=3)
        b2 = c.ax.bar(x + w/2, df["Sales"], width=w, color=Palette.accent, label="Actual", zorder=3)
        for region, bar in zip(df.index, b2):
            c.register_pickable(bar, region)
        c.ax.set_xticks(x)
        c.ax.set_xticklabels(df.index, fontsize=8)
        c.ax.legend(frameon=False, fontsize=8)
        c.style_axes(ylabel="Sales")
        c.fig.tight_layout()
        c.draw()

    def _draw_category_chart(self, state):
        c = self.cat_card.canvas
        c.clear()
        df = self.model.by_category(state.region, state.store)
        y = np.arange(len(df))
        bars = c.ax.barh(y, df["Sales"], color=Palette.series[1], zorder=3)
        for cat, bar in zip(df.index, bars):
            if cat == state.category:
                bar.set_color(Palette.accent_dark)
            c.register_pickable(bar, cat)
        c.ax.set_yticks(y)
        c.ax.set_yticklabels(df.index, fontsize=8)
        c.ax.invert_yaxis()
        c.style_axes(ylabel=None)
        c.fig.tight_layout()
        c.draw()


# ----------------------------------------------------------------------
# 3.2 REGIONAL ANALYSIS PAGE  (Region -> Store drill level)
# ----------------------------------------------------------------------
class RegionalPage(BasePage):
    def build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        self.title_lbl = self.section_title("Regional Analysis")
        root.addWidget(self.title_lbl)

        self.hint_lbl = QLabel()
        self.hint_lbl.setStyleSheet(f"color:{Palette.subtext}; font-size:11px;")
        root.addWidget(self.hint_lbl)

        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(14)
        self.kpi_sales = KpiCard("Region Sales")
        self.kpi_target = KpiCard("Region Target")
        self.kpi_achv = KpiCard("Achievement")
        self.kpi_yoy = KpiCard("YoY Growth")
        for c in (self.kpi_sales, self.kpi_target, self.kpi_achv, self.kpi_yoy):
            kpi_row.addWidget(c)
        root.addLayout(kpi_row)

        grid = QGridLayout()
        grid.setSpacing(14)
        self.store_card = ChartCard("Sales by Store  (click a bar to drill down)")
        self.tva_card = ChartCard("Target vs Actual by Store")
        grid.addWidget(self.store_card, 0, 0)
        grid.addWidget(self.tva_card, 0, 1)
        root.addLayout(grid, stretch=1)

        self.store_card.canvas.bar_clicked.connect(
            lambda label: self.drill_requested.emit("store", label))
        self.tva_card.canvas.bar_clicked.connect(
            lambda label: self.drill_requested.emit("store", label))

    def refresh(self, state):
        region = state.region
        if not region:
            self.hint_lbl.setText("Select a region from the Overview page (or the selector) to drill down into stores.")
        else:
            self.hint_lbl.setText(f"Showing stores within {region}. Click a store to drill down to products.")

        k = self.model.kpis(region, state.store, state.category, state.product)
        self.kpi_sales.update_values(_fmt_money(k["sales"]))
        self.kpi_target.update_values(_fmt_money(k["target"]))
        self.kpi_achv.update_values(f"{k['achievement']:.1f}%",
                                     _fmt_pct(k['achievement'] - 100) + " vs target",
                                     k['achievement'] >= 100)
        self.kpi_yoy.update_values(_fmt_pct(k["yoy"]), "vs prior year", k["yoy"] >= 0)

        self._draw_store_chart(state)
        self._draw_tva(state)

    def _current_region(self, state):
        return state.region or list(self.model.by_region().index)[0]

    def _draw_store_chart(self, state):
        c = self.store_card.canvas
        c.clear()
        region = self._current_region(state)
        df = self.model.by_store(region)
        y = np.arange(len(df))
        bars = c.ax.barh(y, df["Sales"], color=Palette.accent, zorder=3)
        for store, bar in zip(df.index, bars):
            if store == state.store:
                bar.set_color(Palette.accent_dark)
            c.register_pickable(bar, store)
        c.ax.set_yticks(y)
        c.ax.set_yticklabels(df.index, fontsize=8)
        c.ax.invert_yaxis()
        c.style_axes(ylabel=None)
        c.fig.tight_layout()
        c.draw()

    def _draw_tva(self, state):
        c = self.tva_card.canvas
        c.clear()
        region = self._current_region(state)
        df = self.model.by_store(region)
        x = np.arange(len(df))
        w = 0.35
        c.ax.bar(x - w/2, df["Target"], width=w, color=Palette.grid, label="Target", zorder=3)
        b2 = c.ax.bar(x + w/2, df["Sales"], width=w, color=Palette.accent, label="Actual", zorder=3)
        for store, bar in zip(df.index, b2):
            c.register_pickable(bar, store)
        c.ax.set_xticks(x)
        c.ax.set_xticklabels([s.split(" ", 1)[0] for s in df.index], fontsize=7.5, rotation=15)
        c.ax.legend(frameon=False, fontsize=8)
        c.style_axes(ylabel="Sales")
        c.fig.tight_layout()
        c.draw()


# ----------------------------------------------------------------------
# 3.3 STORE / PRODUCT DRILLDOWN PAGE (Store -> Product drill level)
# ----------------------------------------------------------------------
class StorePage(BasePage):
    def build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        self.title_lbl = self.section_title("Store Drill-down")
        root.addWidget(self.title_lbl)

        self.hint_lbl = QLabel()
        self.hint_lbl.setStyleSheet(f"color:{Palette.subtext}; font-size:11px;")
        root.addWidget(self.hint_lbl)

        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(14)
        self.kpi_sales = KpiCard("Store Sales")
        self.kpi_target = KpiCard("Store Target")
        self.kpi_achv = KpiCard("Achievement")
        self.kpi_yoy = KpiCard("YoY Growth")
        for c in (self.kpi_sales, self.kpi_target, self.kpi_achv, self.kpi_yoy):
            kpi_row.addWidget(c)
        root.addLayout(kpi_row)

        grid = QGridLayout()
        grid.setSpacing(14)
        self.product_card = ChartCard("Sales by Product  (click a bar to drill down)")
        self.trend_card = ChartCard("Store Monthly Trend — Current vs Prior Year")
        grid.addWidget(self.product_card, 0, 0)
        grid.addWidget(self.trend_card, 0, 1)
        root.addLayout(grid, stretch=1)

        self.product_card.canvas.bar_clicked.connect(
            lambda label: self.drill_requested.emit("product", label))

    def refresh(self, state):
        region, store = state.region, state.store
        if not store:
            self.hint_lbl.setText("Select a store from Regional Analysis to see its product mix.")
        else:
            self.hint_lbl.setText(f"Showing product performance for {store}. Click a product to see its detail trend.")

        k = self.model.kpis(region, store, state.category, state.product)
        self.kpi_sales.update_values(_fmt_money(k["sales"]))
        self.kpi_target.update_values(_fmt_money(k["target"]))
        self.kpi_achv.update_values(f"{k['achievement']:.1f}%",
                                     _fmt_pct(k['achievement'] - 100) + " vs target",
                                     k['achievement'] >= 100)
        self.kpi_yoy.update_values(_fmt_pct(k["yoy"]), "vs prior year", k["yoy"] >= 0)

        self._draw_product_chart(state)
        self._draw_trend(state)

    def _draw_product_chart(self, state):
        c = self.product_card.canvas
        c.clear()
        df = self.model.by_product(state.region, state.store)
        y = np.arange(len(df))
        bars = c.ax.barh(y, df["Sales"], color=Palette.series[3], zorder=3)
        for prod, bar in zip(df.index, bars):
            if prod == state.product:
                bar.set_color(Palette.accent_dark)
            c.register_pickable(bar, prod)
        c.ax.set_yticks(y)
        c.ax.set_yticklabels(df.index, fontsize=8)
        c.ax.invert_yaxis()
        c.style_axes(ylabel=None)
        c.fig.tight_layout()
        c.draw()

    def _draw_trend(self, state):
        c = self.trend_card.canvas
        c.clear()
        cur, prior = self.model.monthly_trend(state.region, state.store, state.category, state.product)
        x = np.arange(12)
        c.ax.plot(x, prior.values, color=Palette.grid, linewidth=2.2,
                  marker="o", markersize=3, label=f"{self.model.prior_year}")
        c.ax.plot(x, cur.values, color=Palette.accent, linewidth=2.6,
                  marker="o", markersize=3.5, label=f"{self.model.current_year}")
        c.ax.fill_between(x, cur.values, color=Palette.accent, alpha=0.08)
        c.ax.set_xticks(x)
        c.ax.set_xticklabels(MONTH_LABELS, fontsize=7.5)
        c.style_axes(ylabel="Sales")
        c.ax.legend(frameon=False, fontsize=8, loc="upper left")
        c.fig.tight_layout()
        c.draw()


# ----------------------------------------------------------------------
# 3.4 PRODUCT / YoY DEEP-DIVE PAGE
# ----------------------------------------------------------------------
class ProductPage(BasePage):
    def build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        self.title_lbl = self.section_title("Product & YoY Deep Dive")
        root.addWidget(self.title_lbl)

        self.hint_lbl = QLabel()
        self.hint_lbl.setStyleSheet(f"color:{Palette.subtext}; font-size:11px;")
        root.addWidget(self.hint_lbl)

        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(14)
        self.kpi_sales = KpiCard("Sales (selection)")
        self.kpi_yoy = KpiCard("YoY Growth")
        self.kpi_achv = KpiCard("Achievement")
        self.kpi_best = KpiCard("Best Month (current yr)")
        for c in (self.kpi_sales, self.kpi_yoy, self.kpi_achv, self.kpi_best):
            kpi_row.addWidget(c)
        root.addLayout(kpi_row)

        grid = QGridLayout()
        grid.setSpacing(14)
        self.yoy_card = ChartCard("YoY Sales Growth by Category")
        self.gauge_card = ChartCard("Target Achievement (selection)")
        grid.addWidget(self.yoy_card, 0, 0)
        grid.addWidget(self.gauge_card, 0, 1)
        root.addLayout(grid, stretch=1)

        self.yoy_card.canvas.bar_clicked.connect(
            lambda label: self.drill_requested.emit("category", label))

    def refresh(self, state):
        region, store, category, product = state.region, state.store, state.category, state.product
        scope = product or category or store or region or "All Regions"
        self.hint_lbl.setText(f"Current selection scope: {scope}")

        k = self.model.kpis(region, store, category, product)
        self.kpi_sales.update_values(_fmt_money(k["sales"]))
        self.kpi_yoy.update_values(_fmt_pct(k["yoy"]), "vs prior year", k["yoy"] >= 0)
        self.kpi_achv.update_values(f"{k['achievement']:.1f}%",
                                     _fmt_pct(k['achievement'] - 100) + " vs target",
                                     k['achievement'] >= 100)

        cur, _ = self.model.monthly_trend(region, store, category, product)
        best_month_idx = int(cur.values.argmax()) if cur.values.sum() > 0 else 0
        self.kpi_best.update_values(MONTH_LABELS[best_month_idx],
                                     _fmt_money(cur.values[best_month_idx]), True)

        self._draw_yoy_by_category(state)
        self._draw_gauge(k)

    def _draw_yoy_by_category(self, state):
        c = self.yoy_card.canvas
        c.clear()
        cats = self.model.by_category(state.region, state.store).index.tolist()
        growths = []
        for cat in cats:
            k = self.model.kpis(state.region, state.store, cat, None)
            growths.append(k["yoy"])
        colors = [Palette.good if g >= 0 else Palette.bad for g in growths]
        y = np.arange(len(cats))
        bars = c.ax.barh(y, growths, color=colors, zorder=3)
        for cat, bar in zip(cats, bars):
            c.register_pickable(bar, cat)
        c.ax.axvline(0, color=Palette.grid, linewidth=1)
        c.ax.set_yticks(y)
        c.ax.set_yticklabels(cats, fontsize=8)
        c.ax.invert_yaxis()
        c.style_axes(ylabel=None)
        c.ax.set_xlabel("YoY % Growth", color=Palette.subtext, fontsize=8)
        c.fig.tight_layout()
        c.draw()

    def _draw_gauge(self, k):
        """Renders a Power-BI-style semicircular gauge for target achievement."""
        c = self.gauge_card.canvas
        c.clear()
        c.ax.set_aspect("equal")
        achievement = max(0, min(k["achievement"], 150))
        theta_end = np.pi * (1 - min(achievement, 150) / 150)

        # background arc (0-150%)
        bg_theta = np.linspace(0, np.pi, 200)
        c.ax.plot(np.cos(bg_theta), np.sin(bg_theta), color=Palette.grid, linewidth=14,
                  solid_capstyle="round")

        # value arc
        val_theta = np.linspace(np.pi, theta_end, 200)
        color = Palette.good if achievement >= 100 else (Palette.warn if achievement >= 85 else Palette.bad)
        c.ax.plot(np.cos(val_theta), np.sin(val_theta), color=color, linewidth=14,
                  solid_capstyle="round")

        # 100% target tick
        tick_theta = np.pi * (1 - 100 / 150)
        c.ax.plot([np.cos(tick_theta) * 0.85, np.cos(tick_theta) * 1.05],
                   [np.sin(tick_theta) * 0.85, np.sin(tick_theta) * 1.05],
                   color=Palette.text, linewidth=2)

        c.ax.text(0, -0.15, f"{achievement:.1f}%", ha="center", va="center",
                  fontsize=22, fontweight="bold", color=Palette.text)
        c.ax.text(0, -0.42, "of target achieved", ha="center", va="center",
                  fontsize=9, color=Palette.subtext)

        c.ax.set_xlim(-1.2, 1.2)
        c.ax.set_ylim(-0.55, 1.15)
        c.ax.axis("off")
        c.fig.tight_layout()
        c.draw()


# ======================================================================
# 4. MAIN WINDOW / APPLICATION ENTRY POINT
# ======================================================================
# Navigation shell wiring the data model and pages together: sidebar
# navigation, top breadcrumb/filter bar, and the shared cross-filter
# FilterState that every page reads from and writes to via drill-down
# signals (Region -> Store -> Product hierarchy, plus Category).
# ======================================================================

@dataclass
class FilterState:
    region: str = None
    store: str = None
    category: str = None
    product: str = None


NAV_ITEMS = [
    ("Overview", "\U0001F3E0"),
    ("Regional Analysis", "\U0001F5FA"),
    ("Store Drill-down", "\U0001F3EA"),
    ("Product & YoY", "\U0001F4C8"),
]


class NavButton(QPushButton):
    def __init__(self, text, icon_char):
        super().__init__(f"  {icon_char}   {text}")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(44)
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet(f"""
            QPushButton {{
                text-align: left; border: none; color: #C7CBD1;
                background: transparent; padding-left: 18px; border-radius: 6px;
            }}
            QPushButton:hover {{ background: #2A2F38; color: white; }}
            QPushButton:checked {{
                background: {Palette.sidebar_active}; color: white; font-weight: 600;
            }}
        """)


class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Retail Chain Executive Dashboard")
        self.resize(1440, 900)
        self.setStyleSheet(f"background-color:{Palette.bg};")

        self.model = DataModel()
        self.state = FilterState()

        self._build_ui()
        self._refresh_all()

    # ------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_sidebar())

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)
        right_lay.addWidget(self._build_topbar())

        self.stack = QStackedWidget()
        self.pages = {
            "Overview": OverviewPage(self.model),
            "Regional Analysis": RegionalPage(self.model),
            "Store Drill-down": StorePage(self.model),
            "Product & YoY": ProductPage(self.model),
        }
        for page in self.pages.values():
            page.build()
            page.drill_requested.connect(self._on_drill)
            self.stack.addWidget(page)
        right_lay.addWidget(self.stack, stretch=1)

        outer.addWidget(right, stretch=1)

    def _build_sidebar(self):
        panel = QFrame()
        panel.setFixedWidth(230)
        panel.setStyleSheet(f"background-color:{Palette.sidebar};")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(10, 18, 10, 18)
        lay.setSpacing(4)

        brand = QLabel("  RETAIL EXEC\n  DASHBOARD")
        brand.setFont(QFont("Segoe UI", 13, QFont.Bold))
        brand.setStyleSheet("color:white; padding: 6px 8px 18px 8px;")
        lay.addWidget(brand)

        self.nav_buttons = {}
        for name, icon in NAV_ITEMS:
            btn = NavButton(name, icon)
            btn.clicked.connect(lambda _, n=name: self._go_to_page(n))
            lay.addWidget(btn)
            self.nav_buttons[name] = btn
        self.nav_buttons["Overview"].setChecked(True)

        lay.addStretch()

        footer = QLabel(f"Data window: {self.model.prior_year}\u2013{self.model.current_year}\nSynthetic demo data")
        footer.setStyleSheet("color:#7C828C; font-size:10px; padding: 8px;")
        lay.addWidget(footer)
        return panel

    def _build_topbar(self):
        bar = QFrame()
        bar.setFixedHeight(58)
        bar.setStyleSheet(f"background-color:{Palette.card}; border-bottom: 1px solid {Palette.grid};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 8, 20, 8)

        self.breadcrumb = Breadcrumb()
        self.breadcrumb.segment_clicked.connect(self._on_breadcrumb)
        lay.addWidget(self.breadcrumb, stretch=1)

        # Manual region selector (alternate entry point besides clicking bars)
        region_lbl = QLabel("Region:")
        region_lbl.setStyleSheet(f"color:{Palette.subtext}; font-size:11px;")
        self.region_combo = QComboBox()
        self.region_combo.addItem("All Regions")
        self.region_combo.addItems(list(REGIONS.keys()))
        self.region_combo.setFixedWidth(150)
        self.region_combo.currentTextChanged.connect(self._on_region_combo)
        lay.addWidget(region_lbl)
        lay.addWidget(self.region_combo)

        reset_btn = QPushButton("Reset Filters")
        reset_btn.setCursor(Qt.PointingHandCursor)
        reset_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Palette.accent}; color: white; border: none;
                border-radius: 6px; padding: 7px 14px; font-weight: 600; font-size: 11px;
            }}
            QPushButton:hover {{ background-color: {Palette.accent_dark}; }}
        """)
        reset_btn.clicked.connect(self._reset_filters)
        lay.addWidget(reset_btn)
        return bar

    # ------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------
    def _go_to_page(self, name):
        self.stack.setCurrentWidget(self.pages[name])
        for n, btn in self.nav_buttons.items():
            btn.setChecked(n == name)

    # ------------------------------------------------------------
    # Cross-filter / drill-down logic
    # ------------------------------------------------------------
    def _on_drill(self, level, value):
        """Handles a click on any chart bar across any page. Updates the
        shared FilterState hierarchically (Region -> Store -> Product),
        auto-navigates to the next logical drill page (Power BI style),
        and refreshes every page so cross-filtering is visible everywhere."""
        if level == "region":
            self.state = FilterState(region=value)
            self._go_to_page("Regional Analysis")
        elif level == "store":
            self.state = replace(self.state, store=value)
            # infer region from store if not already set
            if not self.state.region:
                for region, stores in REGIONS.items():
                    if value in stores:
                        self.state.region = region
            self._go_to_page("Store Drill-down")
        elif level == "category":
            self.state = replace(self.state, category=value)
        elif level == "product":
            self.state = replace(self.state, product=value)
            self._go_to_page("Product & YoY")

        self._sync_region_combo()
        self._refresh_all()

    def _on_breadcrumb(self, level):
        if level == "all":
            self.state = FilterState()
            self._go_to_page("Overview")
        elif level == "region":
            self.state = FilterState(region=self.state.region)
        elif level == "store":
            self.state = FilterState(region=self.state.region, store=self.state.store)
        elif level == "product":
            pass  # already deepest
        self._sync_region_combo()
        self._refresh_all()

    def _on_region_combo(self, text):
        if text == "All Regions":
            self.state = FilterState()
        else:
            self.state = FilterState(region=text)
        self._refresh_all()

    def _sync_region_combo(self):
        self.region_combo.blockSignals(True)
        self.region_combo.setCurrentText(self.state.region or "All Regions")
        self.region_combo.blockSignals(False)

    def _reset_filters(self):
        self.state = FilterState()
        self._sync_region_combo()
        self._go_to_page("Overview")
        self._refresh_all()

    # ------------------------------------------------------------
    def _refresh_all(self):
        self.breadcrumb.set_path(self.state.region, self.state.store, self.state.product)
        for page in self.pages.values():
            page.refresh(self.state)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = DashboardWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
