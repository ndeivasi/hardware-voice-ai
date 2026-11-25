"""One Audio Model voice loop.

The device setting is ``local`` or ``remote`` (default ``remote``). That one
model receives a captured utterance and produces the spoken reply. This
package does not split the turn into speech-to-text, a language-model call,
and text-to-speech.

Import the modules directly. This package init stays empty so device
configuration can load the gateway settings without building a model.
"""
