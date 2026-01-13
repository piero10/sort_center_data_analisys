import numpy as np
import json
from io import StringIO
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Адаптированный анализ данных", layout="wide")

def load_json_file(uploaded_file):
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, str):
        with open(uploaded_file, "r", encoding="utf-8") as f:
            return json.load(f)
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    return json.load(stringio)

def build_arrivals_df(raw):
    if raw is None or "arrivals" not in raw:
        return None
    arrivals_gate = raw["arrivals"].get("GATE", [])
    df = pd.DataFrame(arrivals_gate)
    if df.empty:
        return df

    if not df.empty and 'unit' in df.columns:
        unit_df = df['unit'].apply(pd.Series)
        df = pd.concat([df.drop('unit', axis=1), unit_df], axis=1)

    if 'arrival_datetime' in df.columns:
        df["arrival_datetime"] = pd.to_datetime(df["arrival_datetime"])
    df = df.sort_values("arrival_datetime").reset_index(drop=True)
    return df

def build_workers_df(raw):
    if raw is None or "workers" not in raw:
        return None

    workers_outer = raw["workers"]
    workers_dict = workers_outer["workers"] if "workers" in workers_outer else workers_outer

    df_workers = (
        pd.DataFrame.from_dict(workers_dict, orient="index")
        .rename_axis("worker_id")
        .reset_index()
    )

    if "stations" in raw:
        stations = raw["stations"]
        df_st = (
            pd.DataFrame.from_dict(stations, orient="index")
            .rename_axis("station_id")
            .reset_index()
        )
        df_st["station_id"] = df_st["station_id"].astype(str)
        station_to_zone = df_st.set_index("station_id")["zone_id"].to_dict()

        if "current_station" in df_workers.columns:
            mask_notna = df_workers["current_station"].notna()
            df_workers.loc[mask_notna, "current_station"] = (
                df_workers.loc[mask_notna, "current_station"]
                .astype(float)
                .astype(int)
                .astype(str)
            )
            df_workers["current_zone"] = df_workers["current_station"].map(station_to_zone).fillna("Простой")
        else:
            df_workers["current_zone"] = "Простой"
    else:
        df_workers["current_zone"] = "Простой"

    if "performance_units" in df_workers.columns:
        perf_expanded = df_workers["performance_units"].apply(pd.Series).add_prefix("perf_")
        df_workers = pd.concat(
            [df_workers.drop(columns=["performance_units"]), perf_expanded], axis=1
        )
    return df_workers


def build_stations_df(raw):
    if raw is None or "stations" not in raw:
        return None

    stations = raw["stations"]
    df_st = (
        pd.DataFrame.from_dict(stations, orient="index")
        .rename_axis("station_id")
        .reset_index()
    )

    # ✅ ПРОВЕРЯЕМ наличие backlog ПЕРЕД обработкой
    if "backlog" in df_st.columns:
        def expand_backlog(backlog_data):
            if not backlog_data or "units" not in backlog_data:
                return pd.Series({"backlog_SORT": 0, "backlog_NONSORT": 0})
            units = backlog_data["units"]
            sort_count = sum(u.get("postings_num", 0) for u in units if u.get("flow_type") == "SORT")
            nonsort_count = sum(u.get("postings_num", 0) for u in units if u.get("flow_type") == "NONSORT")
            return pd.Series({"backlog_SORT": sort_count, "backlog_NONSORT": nonsort_count})

        # 🔥 СНАЧАЛА создаем backlog_units из ОРИГИНАЛЬНОГО backlog
        df_st["backlog_units"] = df_st["backlog"].apply(
            lambda x: len(x.get("units", [])) if x and isinstance(x, dict) else 0
        )

        # ПОТОМ обрабатываем постинги и УДАЛЯЕМ backlog
        bl = df_st["backlog"].apply(expand_backlog)
        df_st = pd.concat([df_st.drop(columns=["backlog"]), bl], axis=1)
        df_st["backlog_total"] = df_st[["backlog_SORT", "backlog_NONSORT"]].sum(axis=1)

    return df_st


st.title("🚀 Анализ входных данных для оптимизации")

# 4 вкладки без "Ст+рабочие"
tab_upload, tab_arrivals, tab_workers, tab_stations_backlog = st.tabs([
    "📁 Загрузка",
    "🚚 Приходы",
    "👷 Рабочие",
    "📊 Станции, Зоны"
])

with tab_upload:
    st.header("📁 Загрузка JSON файла")
    uploaded = st.file_uploader("Загрузите JSON файл", type=["json", "txt"])

    if uploaded is not None:
        try:
            st.session_state["raw_json"] = load_json_file(uploaded)
            st.success("✅ Файл успешно загружен!")
            if st.checkbox("🔍 Показать структуру JSON"):
                st.json({k: f"{str(type(v).__name__)} ({len(v) if hasattr(v, '__len__') else 'N/A'})"
                         for k, v in st.session_state["raw_json"].items()})
        except Exception as e:
            st.error(f"❌ Ошибка загрузки: {e}")
    else:
        st.info("📤 Загрузите файл для анализа данных")

raw = st.session_state.get("raw_json")

# Вкладка Приходы паллет
with tab_arrivals:
    st.header("📦 Приходы паллет (GATE)")
    if raw is None:
        st.warning("⚠️ Сначала загрузите JSON на первой вкладке.")
    else:
        df_arrivals = build_arrivals_df(raw)
        if df_arrivals is None or df_arrivals.empty:
            st.info("ℹ️ Нет данных в arrivals['GATE'].")
        else:
            st.subheader("📋 Таблица прибытий")
            st.dataframe(df_arrivals.head(200), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=["Postings по времени прибытия", "Total postings по типу потока"],
                    vertical_spacing=0.15,  # ← Немного меньше зазор
                    row_heights=[0.5, 0.5]  # ← Основной график больше
                )

                fig.add_trace(
                    go.Scatter(x=df_arrivals["arrival_datetime"], y=df_arrivals["postings_num"],
                               mode='lines+markers', name='Postings', line=dict(color='#1f77b4'),
                               hovertemplate='<b>%{x}</b><br>Postings: %{y:,}<extra></extra>'),
                    row=1, col=1
                )

                if 'flow_type' in df_arrivals.columns:
                    flow_agg = df_arrivals.groupby("flow_type")["postings_num"].sum().reset_index()
                    colors = ['#1f77b4', '#ff7f0e']
                    for i, row in flow_agg.iterrows():
                        fig.add_trace(
                            go.Bar(x=[row["flow_type"]], y=[row["postings_num"]],
                                   marker_color=colors[i % len(colors)], name=row["flow_type"],
                                   hovertemplate='<b>%{x}</b><br>Postings: %{y:,}<extra></extra>'),
                            row=2, col=1
                        )

                fig.update_layout(
                    height=1000,  # ← Было 650 → 850px (шире!)
                    showlegend=False,
                    title_text="Анализ прибытий",
                    template="plotly_white",
                    margin=dict(t=90, b=60, l=60, r=60)  # ← Больше отступы
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                df_cum = df_arrivals.sort_values("arrival_datetime").copy()
                df_cum["cum_postings"] = df_cum["postings_num"].cumsum()
                fig_cum = px.area(df_cum, x="arrival_datetime", y="cum_postings",
                                 title="📊 Кумулятивный объём прибытий", hover_data=["postings_num"])
                fig_cum.update_traces(line_shape="hv")
                fig_cum.update_layout(template="plotly_white")
                st.plotly_chart(fig_cum, use_container_width=True)

# Вкладка Рабочие
with tab_workers:
    st.header("👷 Рабочие")
    if raw is None:
        st.warning("⚠️ Сначала загрузите JSON на первой вкладке.")
    else:
        df_workers = build_workers_df(raw)
        if df_workers is None or df_workers.empty:
            st.info("ℹ️ Нет данных по workers.")
        else:
            st.subheader("📋 Таблица рабочих")
            st.dataframe(df_workers.head(200), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                if "hard_work" in df_workers.columns:
                    hard_counts = df_workers["hard_work"].astype(bool).value_counts().rename(index={True: "Да", False: "Нет"})
                    fig = px.bar(x=hard_counts.index, y=hard_counts.values,
                                title="📊 Способность выполнять тяжёлую работу", labels={'y': 'Количество рабочих'},
                                color=hard_counts.index, color_discrete_map={"Да": "#2ca02c", "Нет": "#d62728"})
                    fig.update_layout(showlegend=False, template="plotly_white")
                    fig.update_traces(texttemplate="%{y}", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if "current_zone" in df_workers.columns:
                    workers_per_zone = df_workers["current_zone"].value_counts(dropna=False).reset_index()
                    workers_per_zone.columns = ["zone_id", "workers_count"]
                    fig = px.bar(workers_per_zone, x="zone_id", y="workers_count", title="📊 Рабочие по зонам", labels={'y': 'Количество рабочих', "workers_count": 'Количество рабочих'})
                    fig.update_layout(template="plotly_white", xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

            with col3:
                perf_cols = [c for c in df_workers.columns if c.startswith("perf_")]
                if perf_cols:
                    perf_means = df_workers[perf_cols].mean().sort_values(ascending=False)
                    fig = px.bar(x=perf_means.index, y=perf_means.values, title="📊 Средняя производительность по зонам", labels={'y': 'Производительность'})
                    fig.update_layout(template="plotly_white", xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

# Вкладка Станции — все графики бэклога
with tab_stations_backlog:
    st.header("🏭 Станции, Зоны")
    if raw is None:
        st.warning("⚠️ Сначала загрузите JSON на первой вкладке.")
    else:
        df_st = build_stations_df(raw)
        if df_st is None or df_st.empty:
            st.info("ℹ️ Нет данных по stations.")
        else:
            st.subheader("Станции - Зоны")
            st.dataframe(df_st.head(200), use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:

                # 1. ВСЕ станции по ПОСТИНГАМ
                if 'backlog_total' in df_st.columns:
                    df_plot = df_st.sort_values("backlog_total", ascending=False)
                    df_plot["name_short"] = df_plot["name"].astype(str).str[:30] + df_plot["name"].astype(str).str[
                        30:].apply(lambda x: "..." if len(x) > 0 else "")
                    fig1 = px.bar(df_plot, y="name_short", x="backlog_total",
                                  title=f"📊 Все станции: Постинги (n={len(df_plot)})",
                                  orientation='h')
                    fig1.update_layout(template="plotly_white", height=650)
                    st.plotly_chart(fig1, use_container_width=True)

                # 2. ВСЕ станции по ЮНИТАМ
                if 'backlog_units' in df_st.columns and df_st["backlog_units"].sum() > 0:
                    df_plot_units = df_st.sort_values("backlog_units", ascending=False)
                    df_plot_units["name_short"] = df_plot_units["name"].astype(str).str[:30] + df_plot_units[
                        "name"].astype(str).str[30:].apply(lambda x: "..." if len(x) > 0 else "")
                    fig2 = px.bar(df_plot_units, y="name_short", x="backlog_units",
                                  title=f"📊 Все станции: Юниты (n={len(df_plot_units)})",
                                  orientation='h')
                    fig2.update_layout(template="plotly_white", height=650)
                    st.plotly_chart(fig2, use_container_width=True)

            with col2:
                if "zone_id" in df_st.columns:
                    stations_per_zone = df_st.groupby("zone_id").agg({
                        "name": "count",
                        "workers_capacity": "sum"
                    }).rename(columns={"name": "stations_count"}).reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="Станций", x=stations_per_zone["zone_id"], y=stations_per_zone["stations_count"], marker_color="#1f77b4"))
                    fig.add_trace(go.Bar(name="Ёмкость рабочих", x=stations_per_zone["zone_id"], y=stations_per_zone["workers_capacity"], marker_color="#ff7f0e"))
                    fig.update_layout(barmode='group', title="📊 Станции и ёмкость по зонам", height=450, template="plotly_white", xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

                # 3. Зоны постинги ✅
                if "zone_id" in df_st.columns and 'backlog_total' in df_st.columns:
                    zone_posts = df_st.groupby("zone_id")["backlog_total"].sum().reset_index()
                    fig3 = px.bar(zone_posts, x="zone_id", y="backlog_total", title="📊 Зоны: Постинги")
                    fig3.update_layout(template="plotly_white", xaxis_tickangle=45, height=450)
                    st.plotly_chart(fig3, use_container_width=True)

                # 4. Зоны юниты ✅
                if "zone_id" in df_st.columns and 'backlog_units' in df_st.columns and df_st["backlog_units"].sum() > 0:
                    zone_units = df_st.groupby("zone_id")["backlog_units"].sum().reset_index()
                    fig4 = px.bar(zone_units, x="zone_id", y="backlog_units", title="📊 Зоны: Юниты")
                    fig4.update_layout(template="plotly_white", xaxis_tickangle=45, height=450)
                    st.plotly_chart(fig4, use_container_width=True)