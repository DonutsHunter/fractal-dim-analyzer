import streamlit as st
import pandas as pd

import scripts.fractal_calculation as sfc
import scripts.image_process as sip
import scripts.data_process as sdp


def fractal_dim() -> None:
    """
    Draw an interface for graphics calculator.

    :return: None
    """

    st.markdown(
        """
        <style>
        div.stButton button {
            height: auto;
            width: 105px;
            padding-top: 20px !important;
            padding-bottom: 20px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    if "trigger_execution" not in st.session_state:
        st.session_state["trigger_execution"] = False

    if 'uploaded' not in st.session_state:
        st.session_state.uploaded = False

    def stop_processing():
        st.session_state.clear()

    def start_processing():
        st.session_state["trigger_execution"] = True

    def uploaded_file():
        st.session_state.uploaded = not st.session_state.uploaded

    upload_col, start_btn_col, stop_btn_col = st.columns([9, 2, 2])

    """
    Имеется баг, что каждый раз когда нажата кнопка,
    состояние обновляется, а значит и состояние кнопок,
    связанных с ним
    """
    with upload_col:
        image_files = st.file_uploader(
            label='## Загрузите файл(ы)',
            type=['png', 'jpg', 'tif'],
            accept_multiple_files=True,
            on_change=uploaded_file
        )

    with start_btn_col:
        st.markdown('##')
        if st.session_state.uploaded == False:
            start_button = st.button(
                label='Рассчитать',
                on_click=start_processing,
                disabled=True
            )
        elif st.session_state.get("uploaded", False):
            start_button = st.button(
                label='Рассчитать',
                on_click=start_processing,
                disabled=False
            )
    
    with stop_btn_col:
        st.markdown('##')
        if st.session_state["trigger_execution"] == False:
            stop_button = st.button(
                label='Отменить',
                disabled=True
            )
        elif st.session_state["trigger_execution"] == True:
            if st.session_state.get("uploaded", True):
                stop_button = st.button(
                    label='Отменить',
                    on_click=stop_processing
                )
            else:
                st.session_state["trigger_execution"] = False

    st.divider()

    point = False
    count = 0

    data_file = './assets/data.xlsx'

    results = []

    if st.session_state["trigger_execution"]:
        progress_bar = st.progress(0, "В ожидании файла(ов) для обработки...")

        if len(image_files) == 1:

            for image_file in image_files:
            # Проверяем, был ли файл загружён
                if image_file is not None:
                    progress_bar.progress(20, "Обработка файла, расчёт фрактальных размерностей...")
                    processed_image = sip.image_separation(image_file)

                    # st.image(processed_image)

                    progress_bar.progress(50, "Построение графика...")
                    box_width_start = 1
                    box_width_end = 100
                    box_width_increment = 1

                    fractal_plot, box_counting, fractal_dimension = sfc.fractal_dim_graphics(
                        image_file=processed_image,
                        box_width_start=box_width_start,
                        box_width_end=box_width_end,
                        box_width_increment=box_width_increment
                    )
                    progress_bar.progress(80, "Построение контура...")
                    progress_bar.progress(100, "Подсчет окончен!")

                    st.divider()

                    result_col1, result_col2 = st.columns([5, 5])

                    with result_col1:
                        st.markdown("### Расчет размерности:")
                        value = list(box_counting.values())[0]
                        for item in value:
                            st.write(item + "\n")

                        value = list(box_counting.values())[-1]
                        for item in value:
                            st.write(item + "\n")

                    st.divider()

                    pic_col1, pic_col2 = st.columns([5, 5])

                    with result_col2:
                        st.markdown(f"""
                        ### Результат:
                        D = {fractal_dimension}""", unsafe_allow_html=True)

                    with pic_col1:
                        st.image('./assets/fd_grid.gif')
                        st.markdown(
                            "<p style=\"text-align: center; font-style: italic\">\
                            Рис. 1. Процесс расчета фрактальной размерности.</p>",
                            unsafe_allow_html=True
                        )

                    with pic_col2:
                        st.pyplot(fractal_plot)
                        st.markdown(
                            "<p style=\"text-align: center; font-style: italic\">\
                            Рис. 2. Линия регрессии по размерам ячеек фрактальной размерности.</p>",
                            unsafe_allow_html=True
                        )
        elif not point:
            for image_file in image_files:
                # Проверяем, был ли файл загружён
                if image_file is not None:
                    progress_bar.progress(
                        value=int(100/len(image_files) * count),
                        text=f"Обработка файла {image_file.name}, расчёт фрактальной размерности..."
                    )
                    processed_image = sip.image_separation(image_file)

                    count += 1

                    box_width_start = 1
                    box_width_end = 100
                    box_width_increment = 1

                    fractal_dimension = sfc.fractal_dim(
                        image_file=processed_image,
                        box_width_start=box_width_start,
                        box_width_end=box_width_end,
                        box_width_increment=box_width_increment
                    )

                    results.append(fractal_dimension)
                    sdp.data_process(results, data_file)

                    if count == len(image_files):
                        progress_bar.progress(100, "Подсчет окончен!")
                        point = True
                        df = pd.read_excel(data_file)
                        st.dataframe(
                            df, 
                            use_container_width=True,
                            hide_index=True,
                        )

            if point:
                with open(data_file, "rb") as file:
                    st.download_button(
                        label="Download xlsx",
                        data=file,
                        file_name="data.xlsx",
                        mime="text/xlsx",
                        disabled=False,
                        use_container_width=True,
                    )
            else:
                with open(data_file, "rb") as file:
                    st.download_button(
                        label="Download .xlsx",
                        data=file,
                        file_name="data.xlsx",
                        mime="text/xlsx",
                        disabled=True,
                        use_container_width=True
                    )
