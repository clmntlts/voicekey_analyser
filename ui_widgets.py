import tkinter as tk
from tkinter import ttk


class Tooltip:
    """Hover tooltip for any widget. ttk has no built-in equivalent."""

    def __init__(self, widget, text, delay_ms=500):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id = None
        self._tipwindow = None
        widget.bind('<Enter>', self._schedule, add='+')
        widget.bind('<Leave>', self._hide, add='+')
        widget.bind('<ButtonPress>', self._hide, add='+')

    def _schedule(self, event=None):
        self._cancel_scheduled()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel_scheduled(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self):
        if self._tipwindow is not None:
            return
        try:
            x = self.widget.winfo_pointerx() + 16
            y = self.widget.winfo_pointery() + 12
        except tk.TclError:
            return

        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left', background='#ffffe0',
                          relief='solid', borderwidth=1, wraplength=320, padx=6, pady=4)
        label.pack()
        self._tipwindow = tw

    def _hide(self, event=None):
        self._cancel_scheduled()
        if self._tipwindow is not None:
            try:
                self._tipwindow.destroy()
            except tk.TclError:
                pass
            self._tipwindow = None


def add_tooltip(widget, text):
    return Tooltip(widget, text)


class CollapsibleFrame(ttk.Frame):
    """A section that can be shown/hidden via a toggle button, for tucking
    away rarely-tuned parameters without a separate tab."""

    def __init__(self, parent, title, collapsed=True, **kwargs):
        super().__init__(parent, **kwargs)
        self.columnconfigure(0, weight=1)
        self._title = title
        self._collapsed = collapsed

        self.toggle_btn = ttk.Button(self, command=self._toggle)
        self.toggle_btn.grid(row=0, column=0, sticky=tk.W)

        self.body = ttk.Frame(self)
        self.body.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self._update_view()

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._update_view()

    def _update_view(self):
        arrow = '▶' if self._collapsed else '▼'
        self.toggle_btn.config(text=f"{arrow} {self._title}")
        if self._collapsed:
            self.body.grid_remove()
        else:
            self.body.grid()
