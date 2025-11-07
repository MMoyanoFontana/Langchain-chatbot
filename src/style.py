from gradio import themes

custom_css = """
/* Custom CSS to adjust Gradio components */
.button-icon{
  width:20px;
  height:20px;
}

.secondary.svelte-034uhq button-icon{
  color:
  revert: invert(1)
}

.wrap.svelte-1kzox3m{
  display: flex;
  flex-direction: column;
}


#chat-list input[type="radio"] {
  display: none !important;
}


#sidebar-logo {
  padding: 10px;
}

#sidebar-logo-container {
 border-bottom: 2px solid var(--border-color-primary);
}

.label-clear-button.svelte-1rvzbk6{
  display: none !important;
  flex-shrink: 2;
}

#logout-btn {
  color: White !important;
  background-color: var(--neutral-500);
  border-color: var(--neutral-500);
}

#logout-btn:hover {
  background-color: var(--neutral-700);
  border-color: var(--neutral-700);
}
.icon-button-wrapper.top-panel.hide-top-corner.svelte-ud4hud {
  display: none !important;
}
"""

gemis_theme = themes.Base(
    neutral_hue="neutral",
    text_size="md",
    radius_size="xxl",
).set(
    prose_header_text_weight="500",
    prose_text_size="md",
    button_border_width="2px",
    button_primary_background_fill="#1861fc",
    button_primary_background_fill_dark="#1861fc",
    button_cancel_background_fill="#DC2626",
    button_cancel_background_fill_hover="#B91C1C",
    button_cancel_background_fill_dark="#DC2626",
    button_cancel_background_fill_hover_dark="#B91C1C",
    button_cancel_border_color="#DC2626",
    button_cancel_border_color_dark="#DC2626",
    button_cancel_text_color="White",
    button_cancel_text_color_dark="White",
    checkbox_background_color="*neutral_400",
    checkbox_background_color_selected="*neutral_200",
    checkbox_label_background_fill="none",
    checkbox_label_background_fill_hover="*neutral_200",
    checkbox_label_background_fill_dark="none",
    checkbox_background_color_dark="*neutral_200",
    checkbox_background_color_selected_dark="*primary_500",
    checkbox_label_background_fill_hover_dark="*neutral_700",
    checkbox_label_background_fill_selected="*neutral_300",
    checkbox_label_background_fill_selected_dark="*neutral_600",
    button_medium_text_weight="500",
    button_large_text_weight="500",
    button_small_text_weight="400",
    button_small_text_size="14px",
    button_medium_text_size="18px",
    checkbox_label_gap="4px",
)
