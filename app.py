import json
from io import StringIO

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


plt.style.use("seaborn-v0_8")

def load_json_file(uploaded_file):
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, str):
        with open(uploaded_file, "r", encoding="utf-8") as f:
            return json.load(f)
    # streamlit UploadedFile
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    return json.load(stringio)


def build_arrivals_df(raw):
    if raw is None or "arrivals" not in raw:
        return None
    arrivals_gate = raw["arrivals"].get("GATE", [])
    df = pd.DataFrame(arrivals_gate)
    if df.empty:
        return df
    df["arrival_datetime"] = pd.to_datetime(df["arrival_datetime"])
    df["cut_off"] = pd.to_datetime(df["cut_off"])
    df["time_to_cutoff_hours"] = (
        df["cut_off"] - df["arrival_datetime"]
    ).dt.total_seconds() / 3600
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

    # current_zone через stations, если есть
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

    # perf_*
    if "performance" in df_workers.columns:
        perf_expanded = df_workers["performance"].apply(pd.Series).add_prefix("perf_")
        df_workers = pd.concat(
            [df_workers.drop(columns=["performance"]), perf_expanded], axis=1
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
    if "backlog_count" in df_st.columns:
        bl = df_st["backlog_count"].apply(pd.Series).fillna(0)
        bl = bl.add_prefix("backlog_")
        df_st = pd.concat([df_st.drop(columns=["backlog_count"]), bl], axis=1)
        df_st["backlog_total"] = df_st[[c for c in df_st.columns if c.startswith("backlog_")]].sum(axis=1)
    return df_st


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Real Data Analysis", layout="wide")

st.title("Анализ реальных данных")

tab_upload, tab_arrivals, tab_workers, tab_stations_backlog, tab_stations_workers = st.tabs(
    [
        "Загрузка файла",
        "Приходы паллет",
        "Рабочие",
        "Станции — бэклоги",
        "Станции — рабочие",
    ]
)

# 1. Загрузка файла

with tab_upload:
    st.header("1. Загрузка JSON файла")
    uploaded = st.file_uploader("Загрузите JSON (data_model / paste)", type=["json", "txt"])
    st.session_state["raw_json"] = load_json_file(uploaded) if uploaded is not None else st.session_state.get("raw_json")

    if st.session_state.get("raw_json") is not None:
        st.success("Файл загружен")
        if st.checkbox("Показать верхний уровень JSON"):
            st.json({k: type(v).__name__ for k, v in st.session_state["raw_json"].items()})
    else:
        st.info("Загрузите файл, чтобы увидеть остальные вкладки в действии.")

raw = st.session_state.get("raw_json")

# 2. Приходы паллет

with tab_arrivals:
    st.header("2. Приходы паллет (GATE)")
    if raw is None:
        st.warning("Сначала загрузите JSON на первой вкладке.")
    else:
        df_arrivals = build_arrivals_df(raw)
        if df_arrivals is None or df_arrivals.empty:
            st.info("Нет данных в arrivals['GATE'].")
        else:
            st.subheader("Таблица")
            st.dataframe(df_arrivals.head(200))

            # временной ряд postings_count
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(df_arrivals["arrival_datetime"], df_arrivals["postings_count"], marker="o")
            ax.set_title("Postings per pallet over time")
            ax.set_xlabel("arrival_datetime")
            ax.set_ylabel("postings_count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # кумулятивный поток
            df_cum = df_arrivals.sort_values("arrival_datetime").copy()
            df_cum["cum_postings"] = df_cum["postings_count"].cumsum()

            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.step(df_cum["arrival_datetime"], df_cum["cum_postings"], where="post")
            ax2.set_title("Cumulative postings over time")
            ax2.set_xlabel("arrival_datetime")
            ax2.set_ylabel("cumulative postings_count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)

            # распределение по flow_type
            flow_agg = df_arrivals.groupby("flow_type")["postings_count"].sum().reset_index()
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            ax3.bar(flow_agg["flow_type"], flow_agg["postings_count"])
            ax3.set_title("Total postings by flow_type")
            ax3.set_xlabel("flow_type")
            ax3.set_ylabel("postings_count")
            plt.tight_layout()
            st.pyplot(fig3)

# 3. Рабочие
with tab_workers:
    st.header("3. Рабочие")
    if raw is None:
        st.warning("Сначала загрузите JSON на первой вкладке.")
    else:
        df_workers = build_workers_df(raw)
        if df_workers is None or df_workers.empty:
            st.info("Нет данных по workers.")
        else:
            st.subheader("Таблица")
            st.dataframe(df_workers.head(200))

            # график по hard_work
            if "hard_work" in df_workers.columns:
                hard_counts = (
                    df_workers["hard_work"]
                    .astype(bool)
                    .value_counts()
                    .rename(index={True: "can_hard", False: "cant_hard"})
                )

                fig, ax = plt.subplots(figsize=(4, 4))
                ax.bar(hard_counts.index, hard_counts.values, color=["tab:green", "tab:red"])
                ax.set_title("Рабочие: могут ли выполнять тяжёлую работу")
                ax.set_xlabel("статус")
                ax.set_ylabel("количество")
                plt.tight_layout()
                st.pyplot(fig)

            # производительность по зонам
            perf_cols = [c for c in df_workers.columns if c.startswith("perf_")]
            if perf_cols:
                st.subheader("Распределение производительности по зонам")

                fig2, ax2 = plt.subplots(figsize=(5, 3))
                df_workers[perf_cols].boxplot(ax=ax2)
                ax2.set_title("Производительность работников по зонам (boxplot)")
                ax2.set_xlabel("зона")
                ax2.set_ylabel("производительность")
                plt.tight_layout()
                st.pyplot(fig2)

            # workers per current_zone
            if "current_zone" in df_workers.columns:
                workers_per_zone = (
                    df_workers["current_zone"]
                    .value_counts(dropna=False)
                    .reset_index()
                )
                workers_per_zone.columns = ["zone_id", "workers_count"]

                fig3, ax3 = plt.subplots(figsize=(6, 4))
                ax3.bar(workers_per_zone["zone_id"].astype(str), workers_per_zone["workers_count"])
                ax3.set_title("Количество рабочих в каждой текущей зоне")
                ax3.set_xlabel("zone_id")
                ax3.set_ylabel("количество рабочих")
                plt.tight_layout()
                st.pyplot(fig3)

# 4. Станции — бэклоги

with tab_stations_backlog:
    st.header("4. Станции — бэклоги")
    if raw is None:
        st.warning("Сначала загрузите JSON на первой вкладке.")
    else:
        df_st = build_stations_df(raw)
        if df_st is None or df_st.empty:
            st.info("Нет данных по stations.")
        else:
            st.subheader("Таблица")
            st.dataframe(df_st.head(200))

            # bar: backlog_total по станциям
            fig, ax = plt.subplots(figsize=(10, 4))
            df_plot = df_st.sort_values("backlog_total", ascending=False)
            ax.bar(df_plot["name"], df_plot["backlog_total"])
            ax.set_title("Бэклог по станциям (всего)")
            ax.set_xlabel("станция")
            ax.set_ylabel("backlog_total")
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(fig)

            # stacked bar SORT / NONSORT по станциям
            bl_cols = [c for c in df_st.columns if c.startswith("backlog_")]
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            bottom = None
            for c in bl_cols:
                values = df_plot[c]
                ax2.bar(df_plot["name"], values, bottom=bottom, label=c)
                bottom = values if bottom is None else bottom + values
            ax2.set_title("Бэклог по станциям по типу потока")
            ax2.set_xlabel("станция")
            ax2.set_ylabel("backlog")
            plt.xticks(rotation=90)
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)

            # backlog по зонам
            if "zone_id" in df_st.columns:
                zone_backlog = (
                    df_st.groupby("zone_id")[bl_cols]
                    .sum()
                    .reset_index()
                )
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                bottom = None
                for c in bl_cols:
                    vals = zone_backlog[c]
                    ax3.bar(zone_backlog["zone_id"], vals, bottom=bottom, label=c)
                    bottom = vals if bottom is None else bottom + vals
                ax3.set_title("Бэклог по зонам")
                ax3.set_xlabel("зона")
                ax3.set_ylabel("backlog")
                ax3.legend()
                plt.tight_layout()
                st.pyplot(fig3)

# 5. Станции —

with tab_stations_workers:
    st.header("5. Станции — рабочие")
    if raw is None:
        st.warning("Сначала загрузите JSON на первой вкладке.")
    else:
        df_st = build_stations_df(raw)
        df_workers = build_workers_df(raw)
        if df_st is None or df_st.empty or df_workers is None or df_workers.empty:
            st.info("Нет данных по stations или workers.")
        else:
            # workers per zone_id (по current_zone)
            if "current_zone" in df_workers.columns:
                workers_per_zone = (
                    df_workers["current_zone"]
                    .value_counts(dropna=False)
                    .reset_index()
                )
                workers_per_zone.columns = ["zone_id", "workers_count"]

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(workers_per_zone["zone_id"].astype(str), workers_per_zone["workers_count"])
                ax.set_title("Количество рабочих по зонам (current_zone)")
                ax.set_xlabel("zone_id")
                ax.set_ylabel("workers_count")
                plt.tight_layout()
                st.pyplot(fig)

            # stations per zone_id
            if "zone_id" in df_st.columns:
                stations_per_zone = (
                    df_st["zone_id"]
                    .value_counts()
                    .reset_index()
                )
                stations_per_zone.columns = ["zone_id", "stations_count"]

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.bar(stations_per_zone["zone_id"], stations_per_zone["stations_count"])
                ax2.set_title("Количество станций по зонам")
                ax2.set_xlabel("zone_id")
                ax2.set_ylabel("stations_count")
                plt.tight_layout()
                st.pyplot(fig2)
