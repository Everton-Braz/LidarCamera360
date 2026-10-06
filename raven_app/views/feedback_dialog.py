"""PyQt6 Fluent dialog for submitting user feedback and bug reports."""
from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

from qfluentwidgets import (
    MessageBoxBase,
    SubtitleLabel,
    LineEdit,
    TextEdit,
    ComboBox,
    CheckBox,
    InfoBar,
)

from raven_app.feedback import (
    gather_diagnostics,
    build_github_issue_url,
    open_github_issue_in_browser,
    send_feedback_webhook,
    export_feedback_bundle,
)


class FeedbackDialog(MessageBoxBase):
    """Dialog for reporting bugs or suggesting features with diagnostics."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.title_label = SubtitleLabel(self.tr("Send Feedback & Bug Report"), self)
        
        self.category_combo = ComboBox(self)
        self.category_combo.addItem(self.tr("Bug Report"), "bug")
        self.category_combo.addItem(self.tr("Feature Request"), "feature")
        self.category_combo.addItem(self.tr("General Feedback"), "question")

        self.title_input = LineEdit(self)
        self.title_input.setPlaceholderText(self.tr("Brief summary of the problem or idea"))

        self.desc_input = TextEdit(self)
        self.desc_input.setPlaceholderText(self.tr("Detailed description, expected behavior, or steps to reproduce..."))
        self.desc_input.setMinimumHeight(120)

        self.contact_input = LineEdit(self)
        self.contact_input.setPlaceholderText(self.tr("Your contact (Email/Discord/GitHub username - optional)"))

        self.include_diag_check = CheckBox(self.tr("Attach non-sensitive system diagnostics (OS, GPU, Python)"), self)
        self.include_diag_check.setChecked(True)

        self.viewLayout.addWidget(self.title_label)
        self.viewLayout.addWidget(QLabel(self.tr("Category:")))
        self.viewLayout.addWidget(self.category_combo)
        self.viewLayout.addWidget(QLabel(self.tr("Title:")))
        self.viewLayout.addWidget(self.title_input)
        self.viewLayout.addWidget(QLabel(self.tr("Details:")))
        self.viewLayout.addWidget(self.desc_input)
        self.viewLayout.addWidget(QLabel(self.tr("Contact (Optional):")))
        self.viewLayout.addWidget(self.contact_input)
        self.viewLayout.addWidget(self.include_diag_check)

        self.yesButton.setText(self.tr("Submit to GitHub"))
        self.cancelButton.setText(self.tr("Cancel"))
        self.yesButton.clicked.connect(self._on_submit_clicked)

        self.widget.setMinimumWidth(480)

    def _on_submit_clicked(self):
        report = self.get_report()
        if not report["title"]:
            return
        send_feedback_webhook(report)
        open_github_issue_in_browser(report)

    def get_report(self) -> dict:
        category = self.category_combo.currentData() or "bug"
        diag = gather_diagnostics() if self.include_diag_check.isChecked() else {}
        return {
            "title": self.title_input.text().strip(),
            "description": self.desc_input.toPlainText().strip(),
            "category": category,
            "contact": self.contact_input.text().strip(),
            "diagnostics": diag,
        }

    def submit_report(self) -> bool:
        report = self.get_report()
        if not report["title"]:
            InfoBar.warning(self.tr("Incomplete"), self.tr("Please provide a title for your report."), parent=self.parent())
            return False

        send_feedback_webhook(report)
        return open_github_issue_in_browser(report)
