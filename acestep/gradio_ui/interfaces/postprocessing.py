"""
Post-Processing UI Module
UI components for audio post-processing effects
"""
import gradio as gr
from acestep.gradio_ui.i18n import t


def create_postprocessing_section():
    """Create the post-processing tab UI."""

    gr.Markdown(f"### {t('postprocessing.subtitle')}")
    gr.Markdown(t("postprocessing.description"))

    # State to hold loaded audio data
    pp_audio_state = gr.State(value=None)

    with gr.Row():
        # Left column: All control accordions
        with gr.Column(scale=1):
            # Input Audio
            with gr.Accordion(t("postprocessing.input_title"), open=True):
                pp_audio_input = gr.Audio(
                    label=t("postprocessing.input_label"),
                    type="filepath"
                )
                pp_load_btn = gr.Button(
                    t("postprocessing.load_btn"),
                    variant="primary"
                )
                pp_load_status = gr.Textbox(
                    label=t("postprocessing.status_label"),
                    interactive=False,
                    lines=2
                )

            # Trim / Cut
            with gr.Accordion(t("postprocessing.trim_title"), open=True):
                with gr.Row():
                    pp_trim_start = gr.Number(
                        label=t("postprocessing.trim_start_label"),
                        value=0,
                        minimum=0
                    )
                    pp_trim_end = gr.Number(
                        label=t("postprocessing.trim_end_label"),
                        value=30,
                        minimum=0
                    )
                pp_trim_btn = gr.Button(t("postprocessing.trim_btn"))

            # Loudness
            with gr.Accordion(t("postprocessing.loudness_title"), open=True):
                pp_gain_db = gr.Slider(
                    label=t("postprocessing.gain_label"),
                    minimum=-20,
                    maximum=20,
                    value=0,
                    step=0.5
                )
                pp_loudness_btn = gr.Button(t("postprocessing.loudness_btn"))
                gr.Markdown("---")
                pp_normalize_db = gr.Slider(
                    label=t("postprocessing.normalize_label"),
                    minimum=-12,
                    maximum=0,
                    value=-1,
                    step=0.5
                )
                pp_normalize_btn = gr.Button(t("postprocessing.normalize_btn"))

            # EQ / Tone
            with gr.Accordion(t("postprocessing.eq_title"), open=True):
                gr.Markdown(f"**{t('postprocessing.bass_title')}**")
                with gr.Row():
                    pp_bass_boost_db = gr.Slider(
                        label=t("postprocessing.boost_label"),
                        minimum=0,
                        maximum=12,
                        value=6,
                        step=0.5
                    )
                    pp_bass_cutoff = gr.Slider(
                        label=t("postprocessing.cutoff_label"),
                        minimum=60,
                        maximum=300,
                        value=150,
                        step=10
                    )
                pp_bass_btn = gr.Button(t("postprocessing.bass_btn"))

                gr.Markdown(f"**{t('postprocessing.treble_title')}**")
                with gr.Row():
                    pp_treble_boost_db = gr.Slider(
                        label=t("postprocessing.boost_label"),
                        minimum=0,
                        maximum=12,
                        value=6,
                        step=0.5
                    )
                    pp_treble_cutoff = gr.Slider(
                        label=t("postprocessing.cutoff_label"),
                        minimum=2000,
                        maximum=10000,
                        value=4000,
                        step=100
                    )
                pp_treble_btn = gr.Button(t("postprocessing.treble_btn"))

            # Fades
            with gr.Accordion(t("postprocessing.fades_title"), open=True):
                with gr.Row():
                    pp_fade_in_duration = gr.Slider(
                        label=t("postprocessing.fade_in_label"),
                        minimum=0,
                        maximum=10,
                        value=2,
                        step=0.1
                    )
                    pp_fade_in_btn = gr.Button(t("postprocessing.fade_in_btn"))
                with gr.Row():
                    pp_fade_out_duration = gr.Slider(
                        label=t("postprocessing.fade_out_label"),
                        minimum=0,
                        maximum=10,
                        value=2,
                        step=0.1
                    )
                    pp_fade_out_btn = gr.Button(t("postprocessing.fade_out_btn"))

            # Speed
            with gr.Accordion(t("postprocessing.speed_title"), open=True):
                pp_speed_factor = gr.Slider(
                    label=t("postprocessing.speed_label"),
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.05
                )
                pp_speed_btn = gr.Button(t("postprocessing.speed_btn"))

            # AudioSR Super-Resolution
            with gr.Accordion(t("postprocessing.audiosr_title"), open=False):
                gr.Markdown(t("postprocessing.audiosr_description"))

                # Model selection
                pp_audiosr_model = gr.Dropdown(
                    label=t("postprocessing.audiosr_model_label"),
                    choices=["basic", "speech"],
                    value="basic",
                    info=t("postprocessing.audiosr_model_info")
                )

                # Quality settings
                gr.Markdown(f"**{t('postprocessing.audiosr_quality_title')}**")
                with gr.Row():
                    pp_audiosr_ddim_steps = gr.Slider(
                        label=t("postprocessing.audiosr_ddim_steps_label"),
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=5,
                        info=t("postprocessing.audiosr_ddim_steps_info")
                    )
                    pp_audiosr_guidance_scale = gr.Slider(
                        label=t("postprocessing.audiosr_guidance_scale_label"),
                        minimum=1.0,
                        maximum=10.0,
                        value=3.5,
                        step=0.1,
                        info=t("postprocessing.audiosr_guidance_scale_info")
                    )

                with gr.Row():
                    pp_audiosr_seed = gr.Number(
                        label=t("postprocessing.audiosr_seed_label"),
                        value=-1,
                        precision=0,
                        info=t("postprocessing.audiosr_seed_info")
                    )
                    pp_audiosr_device = gr.Dropdown(
                        label=t("postprocessing.audiosr_device_label"),
                        choices=["auto", "cuda", "cpu"],
                        value="auto",
                        info=t("postprocessing.audiosr_device_info")
                    )

                # Preprocessing
                gr.Markdown(f"**{t('postprocessing.audiosr_preprocess_title')}**")
                with gr.Row():
                    pp_audiosr_prefilter = gr.Checkbox(
                        label=t("postprocessing.audiosr_prefilter_label"),
                        value=True,
                        info=t("postprocessing.audiosr_prefilter_info")
                    )
                    pp_audiosr_prefilter_cutoff = gr.Slider(
                        label=t("postprocessing.audiosr_prefilter_cutoff_label"),
                        minimum=4000,
                        maximum=20000,
                        value=12000,
                        step=500,
                        info=t("postprocessing.audiosr_prefilter_cutoff_info")
                    )

                # Long audio processing
                gr.Markdown(f"**{t('postprocessing.audiosr_chunking_title')}**")
                with gr.Row():
                    pp_audiosr_chunk_duration = gr.Slider(
                        label=t("postprocessing.audiosr_chunk_duration_label"),
                        minimum=2.56,
                        maximum=10.24,
                        value=10.24,
                        step=0.01,
                        info=t("postprocessing.audiosr_chunk_duration_info")
                    )
                    pp_audiosr_overlap_duration = gr.Slider(
                        label=t("postprocessing.audiosr_overlap_duration_label"),
                        minimum=0.5,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                        info=t("postprocessing.audiosr_overlap_duration_info")
                    )

                # Output settings
                gr.Markdown(f"**{t('postprocessing.audiosr_output_title')}**")
                with gr.Row():
                    pp_audiosr_normalize = gr.Checkbox(
                        label=t("postprocessing.audiosr_normalize_label"),
                        value=True,
                        info=t("postprocessing.audiosr_normalize_info")
                    )
                    pp_audiosr_output_format = gr.Dropdown(
                        label=t("postprocessing.audiosr_output_format_label"),
                        choices=["wav", "mp3", "flac"],
                        value="wav",
                        info=t("postprocessing.audiosr_output_format_info")
                    )

                # Action buttons
                with gr.Row():
                    pp_audiosr_btn = gr.Button(
                        t("postprocessing.audiosr_btn"),
                        variant="primary"
                    )
                    pp_audiosr_unload_btn = gr.Button(
                        t("postprocessing.audiosr_unload_btn"),
                        variant="secondary"
                    )

        # Right column: Output
        with gr.Column(scale=1):
            with gr.Accordion(t("postprocessing.output_title"), open=True):
                pp_output_audio = gr.Audio(
                    label=t("postprocessing.output_label"),
                    interactive=False
                )
                pp_output_status = gr.Textbox(
                    label=t("postprocessing.output_status_label"),
                    interactive=False,
                    lines=2
                )

    # Return all components in a dictionary
    return {
        "pp_audio_state": pp_audio_state,
        "pp_audio_input": pp_audio_input,
        "pp_load_btn": pp_load_btn,
        "pp_load_status": pp_load_status,
        "pp_trim_start": pp_trim_start,
        "pp_trim_end": pp_trim_end,
        "pp_trim_btn": pp_trim_btn,
        "pp_gain_db": pp_gain_db,
        "pp_loudness_btn": pp_loudness_btn,
        "pp_normalize_db": pp_normalize_db,
        "pp_normalize_btn": pp_normalize_btn,
        "pp_bass_boost_db": pp_bass_boost_db,
        "pp_bass_cutoff": pp_bass_cutoff,
        "pp_bass_btn": pp_bass_btn,
        "pp_treble_boost_db": pp_treble_boost_db,
        "pp_treble_cutoff": pp_treble_cutoff,
        "pp_treble_btn": pp_treble_btn,
        "pp_fade_in_duration": pp_fade_in_duration,
        "pp_fade_in_btn": pp_fade_in_btn,
        "pp_fade_out_duration": pp_fade_out_duration,
        "pp_fade_out_btn": pp_fade_out_btn,
        "pp_speed_factor": pp_speed_factor,
        "pp_speed_btn": pp_speed_btn,
        "pp_output_audio": pp_output_audio,
        "pp_output_status": pp_output_status,
        # AudioSR components
        "pp_audiosr_model": pp_audiosr_model,
        "pp_audiosr_ddim_steps": pp_audiosr_ddim_steps,
        "pp_audiosr_guidance_scale": pp_audiosr_guidance_scale,
        "pp_audiosr_seed": pp_audiosr_seed,
        "pp_audiosr_device": pp_audiosr_device,
        "pp_audiosr_prefilter": pp_audiosr_prefilter,
        "pp_audiosr_prefilter_cutoff": pp_audiosr_prefilter_cutoff,
        "pp_audiosr_chunk_duration": pp_audiosr_chunk_duration,
        "pp_audiosr_overlap_duration": pp_audiosr_overlap_duration,
        "pp_audiosr_normalize": pp_audiosr_normalize,
        "pp_audiosr_output_format": pp_audiosr_output_format,
        "pp_audiosr_btn": pp_audiosr_btn,
        "pp_audiosr_unload_btn": pp_audiosr_unload_btn,
    }
