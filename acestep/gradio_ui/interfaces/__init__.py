"""
Gradio UI Components Module
Contains all Gradio interface component definitions and layouts
"""
import gradio as gr
from acestep.gradio_ui.i18n import get_i18n, t
from acestep.gradio_ui.interfaces.dataset import create_dataset_section
from acestep.gradio_ui.interfaces.generation import create_generation_section
from acestep.gradio_ui.interfaces.result import create_results_section
from acestep.gradio_ui.interfaces.postprocessing import create_postprocessing_section
from acestep.gradio_ui.interfaces.training import create_training_section
from acestep.gradio_ui.events import setup_event_handlers, setup_training_event_handlers
from acestep.gradio_ui.events.postprocessing_handlers import setup_postprocessing_event_handlers


def create_gradio_interface(dit_handler, llm_handler, dataset_handler, init_params=None, language='en') -> gr.Blocks:
    """
    Create Gradio interface
    
    Args:
        dit_handler: DiT handler instance
        llm_handler: LM handler instance
        dataset_handler: Dataset handler instance
        init_params: Dictionary containing initialization parameters and state.
                    If None, service will not be pre-initialized.
        language: UI language code ('en', 'zh', 'ja', default: 'en')
        
    Returns:
        Gradio Blocks instance
    """
    # Initialize i18n with selected language
    i18n = get_i18n(language)
    
    with gr.Blocks(
        title=t("app.title"),
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.neutral,
        ),
        css="""
        .green-btn {
            background: linear-gradient(to bottom right, #2ecc71, #27ae60) !important;
            color: white !important;
            border: none !important;
        }
        .green-btn:hover {
            background: linear-gradient(to bottom right, #27ae60, #219651) !important;
        }
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .lm-hints-row {
            align-items: stretch;
        }
        .lm-hints-col {
            display: flex;
        }
        .lm-hints-col > div {
            flex: 1;
            display: flex;
        }
        .lm-hints-btn button {
            height: 100%;
            width: 100%;
        }
        /* Position Audio time labels lower to avoid scrollbar overlap */
        .component-wrapper > .timestamps {
            transform: translateY(15px);
        }
        """,
    ) as demo:

        gr.HTML(f"""
        <div class="main-header">
            <h1>{t("app.title")}</h1>
            <p>{t("app.subtitle")}</p>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("Generation"):
                # Dataset Explorer Section
                dataset_section = create_dataset_section(dataset_handler)

                # Generation Section (pass init_params and language to support pre-initialization)
                generation_section = create_generation_section(dit_handler, llm_handler, init_params=init_params, language=language)

                # Results Section
                results_section = create_results_section(dit_handler)

            with gr.Tab("Training"):
                # Training Section (LoRA training and dataset builder)
                # Pass init_params to support hiding in service mode
                training_section = create_training_section(dit_handler, llm_handler, init_params=init_params)

            with gr.Tab("Post-Processing"):
                # Post-Processing Section
                postprocessing_section = create_postprocessing_section()
        
        # Connect event handlers
        setup_event_handlers(demo, dit_handler, llm_handler, dataset_handler, dataset_section, generation_section, results_section)

        # Connect post-processing event handlers
        setup_postprocessing_event_handlers(demo, postprocessing_section)

        # Connect training event handlers
        setup_training_event_handlers(demo, dit_handler, llm_handler, training_section)
    
    return demo
