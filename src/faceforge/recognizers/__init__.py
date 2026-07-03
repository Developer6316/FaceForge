"""
Face recognizers module - provides base Recognizer class and recognizer implementations
"""
from .sv_classifier import Recognizer, SimpleVerifier

__all__ = ["Recognizer", "SimpleVerifier"]
