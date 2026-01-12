import numpy as np
import json
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

plt.style.use("default")


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
            df_workers["current_zone"] = df_workers["current_station"].map(station_to_zone)
        else:
            df_workers["current_zone"] = pd.NA
    else:
        df_workers["current_zone"] = pd.NA

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

    if "backlog" in df_st.columns:
        def expand_backlog(backlog_data):
            if not backlog_data or "units" not in backlog_data:
                return pd.Series({"backlog_SORT": 0, "backlog_NONSORT": 0})
            units = backlog_data["units"]
            sort_count = sum(u.get("postings_num", 0) for u in units if u.get("flow_type") == "SORT")
            nonsort_count = sum(u.get("postings_num", 0) for u in units if u.get("flow_type") == "NONSORT")
            return pd.Series({"backlog_SORT": sort_count, "backlog_NONSORT": nonsort_count})

        bl = df_st["backlog"].apply(expand_backlog)
        df_st = pd.concat([df_st.drop(columns=["backlog"]), bl], axis=1)
        df_st["backlog_total"] = df_st[["backlog_SORT", "backlog_NONSORT"]].sum(axis=1)

    return df_st


st.set_page_config(page_title="Адаптированный анализ данных", layout="wide")

st.title("Анализ данных склада")

tab_upload, tab_arrivals, tab_workers, tab_stations_backlog, tab_stations_workers = st.tabs([
    "Загрузка файла",
    "Приходы паллет",
    "Рабочие",
    "Станции — бэклоги",
    "Станции — рабочие",
])

with tab_upload:
    st.header("Загрузка JSON файла")
    uploaded = st.file_uploader("Загрузите JSON файл", type=["json", "txt"])

    if uploaded is not None:
        try:
            st.session_state["raw_json"] = load_json_file(uploaded)
            st.success("Файл успешно загружен!")

            if st.checkbox("Показать структуру JSON"):
                st.json({k: str(type(v).__name__) + f" ({len(v) if hasattr(v, '__len__') else 'N/A'})"
                         for k, v in st.session_state["raw_json"].items()})
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
    else:
        st.info("Загрузите файл для анализа данных")

raw = st.session_state.get("raw_json")

with tab_arrivals:
    st.header("Приходы паллет (GATE)")
    if raw is None:
        st.warning("Сначала загрузите JSON на первой вкладке.")
    else:
        df_arrivals = build_arrivals_df(raw)
        if df_arrivals is None or df_arrivals.empty:
            st.info("Нет данных в arrivals['GATE'].")
        else:
            st.subheader("Таблица прибытий")
            st.dataframe(df_arrivals.head(200), use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(df_arrivals["arrival_datetime"], df_arrivals["postings_num"],
                        marker="o", markersize=3, alpha=0.7)
                ax.set_title("Postings по времени прибытия")
                ax.set_xlabel("Время прибытия")
                ax.set_ylabel("Количество postings")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                df_cum = df_arrivals.sort_values("arrival_datetime").copy()
                df_cum["cum_postings"] = df_cum["postings_num"].cumsum()

                fig2, ax2 = plt.subplots(figsize=(8, 4))
                ax2.step(df_cum["arrival_datetime"], df_cum["cum_postings"], where="post", linewidth=2)
                ax2.set_title("Кумулятивные postings")
                ax2.set_xlabel("Время прибытия")
                ax2.set_ylabel("Кумулятивное количество")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig2)

            if 'flow_type' in df_arrivals.columns:
                flow_agg = df_arrivals.groupby("flow_type")["postings_num"].sum().reset_index()
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                colors = ['#1f77b4', '#ff7f0e']
                ax3.bar(flow_agg["flow_type"], flow_agg["postings_num"], color=colors)
                ax3.set_title("Total postings по типу потока")  
                ax3.set_ylabel("Количество postings")
                plt.tight_layout()
                st.pyplot(fig3)

with tab_workers:
    st.header("Рабочие")
    if raw is None:
        st.warning("Сначала загрузите JSON на первой вкладке.")
    else:
        df_workers = build_workers_df(raw)
        if df_workers is None or df_workers.empty:
            st.info("Нет данных по workers.")
        else:
            st.subheader("Таблица рабочих")
            st.dataframe(df_workers.head(200), use_container_width=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                if "hard_work" in df_workers.columns:
                    hard_counts = (
                        df_workers["hard_work"]
                        .astype(bool)
                        .value_counts()
                        .rename(index={True: "Могут тяжёлую", False: "Не могут тяжёлую"})
                    )
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ['#2ca02c', '#d62728']
                    ax.bar(hard_counts.index, hard_counts.values, color=colors)
                    ax.set_title("Способность к тяжёлой работе")
                    ax.set_ylabel("Количество рабочих")
                    for i, v in enumerate(hard_counts.values):
                        ax.text(i, v + 1, str(v), ha='center', va='bottom')
                    plt.tight_layout()
                    st.pyplot(fig)

            with col2:
                if "current_zone" in df_workers.columns:
                    workers_per_zone = (
                        df_workers["current_zone"]
                        .value_counts(dropna=False)
                        .reset_index()
                    )
                    workers_per_zone.columns = ["zone_id", "workers_count"]
                    fig3, ax3 = plt.subplots(figsize=(6, 4))
                    ax3.bar(workers_per_zone["zone_id"].astype(str), workers_per_zone["workers_count"])
                    ax3.set_title("Рабочие по зонам")
                    ax3.set_ylabel("Количество")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig3)

            with col3:
                perf_cols = [c for c in df_workers.columns if c.startswith("perf_")]
                if perf_cols:
                    perf_means = df_workers[perf_cols].mean().sort_values(ascending=False)
                    fig4, ax4 = plt.subplots(figsize=(6, 4))
                    perf_means.plot(kind='bar', ax=ax4)
                    ax4.set_title("Средняя производительность по зонам")
                    ax4.set_ylabel("Единиц/час")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig4)

with tab_stations_backlog:
    st.header("Станции")
    if raw is None:
        st.warning("Сначала загрузите JSON на первой вкладке.")
    else:
        df_st = build_stations_df(raw)
        if df_st is None or df_st.empty:
            st.info("Нет данных по stations.")
        else:
            st.subheader("Таблица станций")
            st.dataframe(df_st.head(200), use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                if 'backlog_total' in df_st.columns:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    df_plot = df_st.sort_values("backlog_total", ascending=False)
                    ax.barh(range(len(df_plot)), df_plot["backlog_total"])
                    ax.set_yticks(range(len(df_plot)))
                    ax.set_yticklabels([f"{row['name'][:20]}..." if len(str(row['name'])) > 20 else str(row['name'])
                                        for _, row in df_plot.iterrows()], fontsize=8)
                    ax.set_title("Бэклог по станциям")
                    ax.set_xlabel("Количество postings")
                    plt.tight_layout()
                    st.pyplot(fig)

            with col2:
                if "zone_id" in df_st.columns:
                    stations_per_zone = df_st.groupby("zone_id").agg({
                        "name": "count",
                        "workers_capacity": "sum"
                    }).rename(columns={"name": "stations_count"})
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    x = np.arange(len(stations_per_zone))
                    width = 0.35
                    ax2.bar(x, stations_per_zone["stations_count"], width, label="Станций")
                    ax2.bar(x + width, stations_per_zone["workers_capacity"], width, label="Суммарная емкость")
                    ax2.set_title("Станции и ёмкость по зонам")
                    ax2.set_xticks(x + width / 2)
                    ax2.set_xticklabels(stations_per_zone.index)
                    ax2.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig2)

with tab_stations_workers:
    st.header("Станции и рабочие")
    if raw is None:
        st.warning("Сначала загрузите JSON на первой вкладке.")
    else:
        df_st = build_stations_df(raw)
        df_workers = build_workers_df(raw)
        if df_st is None or df_st.empty or df_workers is None or df_workers.empty:
            st.info("Недостаточно данных.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                if "current_zone" in df_workers.columns:
                    workers_per_zone = df_workers["current_zone"].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 5))
                    workers_per_zone.plot(kind='bar', ax=ax)
                    ax.set_title("Распределение рабочих по зонам")
                    ax.set_ylabel("Количество рабочих")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

            with col2:
                if "zone_id" in df_st.columns:
                    stations_per_zone = df_st.groupby("zone_id").agg({
                        "name": "count",
                        "workers_capacity": "sum"
                    }).rename(columns={"name": "stations_count"})
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    x = np.arange(len(stations_per_zone))
                    width = 0.35
                    ax2.bar(x, stations_per_zone["stations_count"], width, label="Станций")
                    ax2.bar(x + width, stations_per_zone["workers_capacity"], width, label="Суммарная емкость")
                    ax2.set_title("Станции и ёмкость по зонам")
                    ax2.set_xticks(x + width / 2)
                    ax2.set_xticklabels(stations_per_zone.index)
                    ax2.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig2)
