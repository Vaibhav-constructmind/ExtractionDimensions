"""Drawing dimension extraction pipeline.

Combines Azure Document Intelligence (OCR/layout) with Claude Sonnet-5
(vision + reasoning, served via an Azure AI Foundry deployment) to turn an
uploaded architectural PDF into structured JSON describing every drawing on
the sheet and every dimension found on each drawing.
"""
