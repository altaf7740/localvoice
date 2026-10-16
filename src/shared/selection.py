import sys
import time

if sys.platform == "darwin":
    from AppKit import NSPasteboard, NSPasteboardTypeString
    from Quartz import (
        CGEventCreateKeyboardEvent,
        CGEventPost,
        CGEventSetFlags,
        kCGEventFlagMaskCommand,
        kCGHIDEventTap,
    )
else:
    import pyperclip
    from pynput.keyboard import Controller, Key

C_KEYCODE = 8
COPY_TIMEOUT = 0.5  # apps answer ⌘C within a few tens of ms; nothing by then = no selection
SENTINEL = "⁣localvoice⁣"  # invisible marker to detect whether Ctrl+C copied anything


def selected_text() -> str:
    """Copy the focused app's selection and return it, then restore the user's clipboard."""
    if sys.platform == "darwin":
        return _copy_mac()
    return _copy_generic()


def _copy_mac() -> str:
    pb = NSPasteboard.generalPasteboard()
    previous = pb.stringForType_(NSPasteboardTypeString)
    count = pb.changeCount()
    for down in (True, False):
        event = CGEventCreateKeyboardEvent(None, C_KEYCODE, down)
        CGEventSetFlags(event, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, event)
    deadline = time.time() + COPY_TIMEOUT
    while pb.changeCount() == count and time.time() < deadline:
        time.sleep(0.01)
    if pb.changeCount() == count:
        return ""
    text = pb.stringForType_(NSPasteboardTypeString) or ""
    if previous is not None:
        pb.clearContents()
        pb.setString_forType_(previous, NSPasteboardTypeString)
    return text


def _copy_generic() -> str:
    try:
        previous = pyperclip.paste()
    except pyperclip.PyperclipException:
        previous = None
    pyperclip.copy(SENTINEL)
    keyboard = Controller()
    with keyboard.pressed(Key.ctrl):
        keyboard.tap("c")
    deadline = time.time() + COPY_TIMEOUT
    text = SENTINEL
    while text == SENTINEL and time.time() < deadline:
        time.sleep(0.01)
        text = pyperclip.paste()
    pyperclip.copy(previous or "")
    return "" if text == SENTINEL else text
