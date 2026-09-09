import unittest

from raven_app.i18n import available_languages, get_language, set_language, tr


class I18nTests(unittest.TestCase):
    def tearDown(self):
        set_language("en")

    def test_all_supported_catalogs_load(self):
        self.assertEqual({code for code, _ in available_languages()}, {"en", "pt-BR", "es", "fr", "de", "zh-CN", "ja"})
        for code, _ in available_languages():
            self.assertEqual(set_language(code), code)
            self.assertTrue(tr("Settings & Personalization"))

    def test_portuguese_translates_operational_labels(self):
        set_language("pt-BR")
        self.assertEqual(tr("Settings & Personalization"), "Configurações e personalização")
        self.assertEqual(tr("Reconstruction-based colorization"), "Colorização baseada em reconstrução")

    def test_unknown_language_falls_back_to_english(self):
        self.assertEqual(set_language("xx"), "en")
        self.assertEqual(get_language(), "en")
        self.assertEqual(tr("Settings & Personalization"), "Settings & Personalization")


if __name__ == "__main__":
    unittest.main()
