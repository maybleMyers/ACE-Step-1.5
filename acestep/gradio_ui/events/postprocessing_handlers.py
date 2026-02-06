"""
Post-Processing Event Handlers Module
Connects UI components to DSP processing functions
"""
from acestep.gradio_ui.postprocessing_utils import (
    load_audio_for_processing,
    trim_audio,
    adjust_loudness,
    normalize_audio,
    apply_bass_boost,
    apply_treble_boost,
    apply_fade_in,
    apply_fade_out,
    change_speed,
)


def setup_postprocessing_event_handlers(demo, postprocessing_section):
    """
    Setup event handlers for post-processing tab.

    Args:
        demo: Gradio demo instance
        postprocessing_section: Dictionary of post-processing UI components
    """

    # Load audio
    postprocessing_section["pp_load_btn"].click(
        fn=load_audio_for_processing,
        inputs=[postprocessing_section["pp_audio_input"]],
        outputs=[
            postprocessing_section["pp_audio_state"],
            postprocessing_section["pp_output_audio"],
            postprocessing_section["pp_load_status"],
        ]
    )

    # Trim audio
    postprocessing_section["pp_trim_btn"].click(
        fn=trim_audio,
        inputs=[
            postprocessing_section["pp_audio_state"],
            postprocessing_section["pp_trim_start"],
            postprocessing_section["pp_trim_end"],
        ],
        outputs=[
            postprocessing_section["pp_output_audio"],
            postprocessing_section["pp_output_status"],
        ]
    )

    # Adjust loudness
    postprocessing_section["pp_loudness_btn"].click(
        fn=adjust_loudness,
        inputs=[
            postprocessing_section["pp_audio_state"],
            postprocessing_section["pp_gain_db"],
        ],
        outputs=[
            postprocessing_section["pp_output_audio"],
            postprocessing_section["pp_output_status"],
        ]
    )

    # Normalize audio
    postprocessing_section["pp_normalize_btn"].click(
        fn=normalize_audio,
        inputs=[
            postprocessing_section["pp_audio_state"],
            postprocessing_section["pp_normalize_db"],
        ],
        outputs=[
            postprocessing_section["pp_output_audio"],
            postprocessing_section["pp_output_status"],
        ]
    )

    # Apply bass boost
    postprocessing_section["pp_bass_btn"].click(
        fn=apply_bass_boost,
        inputs=[
            postprocessing_section["pp_audio_state"],
            postprocessing_section["pp_bass_boost_db"],
            postprocessing_section["pp_bass_cutoff"],
        ],
        outputs=[
            postprocessing_section["pp_output_audio"],
            postprocessing_section["pp_output_status"],
        ]
    )

    # Apply treble boost
    postprocessing_section["pp_treble_btn"].click(
        fn=apply_treble_boost,
        inputs=[
            postprocessing_section["pp_audio_state"],
            postprocessing_section["pp_treble_boost_db"],
            postprocessing_section["pp_treble_cutoff"],
        ],
        outputs=[
            postprocessing_section["pp_output_audio"],
            postprocessing_section["pp_output_status"],
        ]
    )

    # Apply fade-in
    postprocessing_section["pp_fade_in_btn"].click(
        fn=apply_fade_in,
        inputs=[
            postprocessing_section["pp_audio_state"],
            postprocessing_section["pp_fade_in_duration"],
        ],
        outputs=[
            postprocessing_section["pp_output_audio"],
            postprocessing_section["pp_output_status"],
        ]
    )

    # Apply fade-out
    postprocessing_section["pp_fade_out_btn"].click(
        fn=apply_fade_out,
        inputs=[
            postprocessing_section["pp_audio_state"],
            postprocessing_section["pp_fade_out_duration"],
        ],
        outputs=[
            postprocessing_section["pp_output_audio"],
            postprocessing_section["pp_output_status"],
        ]
    )

    # Change speed
    postprocessing_section["pp_speed_btn"].click(
        fn=change_speed,
        inputs=[
            postprocessing_section["pp_audio_state"],
            postprocessing_section["pp_speed_factor"],
        ],
        outputs=[
            postprocessing_section["pp_output_audio"],
            postprocessing_section["pp_output_status"],
        ]
    )
