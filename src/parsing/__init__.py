"""Document parsing: extracting structured content from standards PDFs."""

from .pdf_to_structure import (
    build_structure,
    detect_doc_type,
    parse_pdf,
    process_batch,
    standard_id_from_filename,
)
from .patterns import DOC_TYPE_CN_GB, DOC_TYPE_ENTERPRISE, DOC_TYPE_IEC, DOC_TYPE_ISO

__all__ = [
    "build_structure",
    "detect_doc_type",
    "parse_pdf",
    "process_batch",
    "standard_id_from_filename",
    "DOC_TYPE_CN_GB",
    "DOC_TYPE_ISO",
    "DOC_TYPE_IEC",
    "DOC_TYPE_ENTERPRISE",
]
