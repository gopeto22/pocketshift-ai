"""Capture submission screenshots with Playwright (waits for real render).

Usage: python capture_screens.py  (app must be running on :8521)
"""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("outputs/screenshots")
OUT.mkdir(parents=True, exist_ok=True)
import os as _os

URL = _os.environ.get("POCKETSHIFT_URL", "http://localhost:8521/")


def wait_render(page, text: str, timeout=30000):
    page.wait_for_selector(f"text={text}", timeout=timeout)
    time.sleep(1.5)  # let charts paint


def click_nav(page, label: str):
    page.get_by_text(label, exact=False).first.click()
    time.sleep(2.5)


def main():
    import os

    shell = os.path.expanduser(
        "~/Library/Caches/ms-playwright/chromium_headless_shell-1208/"
        "chrome-headless-shell-mac-arm64/chrome-headless-shell"
    )
    launch_kwargs = {"executable_path": shell} if os.path.exists(shell) else {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(**launch_kwargs)
        page = browser.new_page(viewport={"width": 1600, "height": 1500})
        page.goto(URL, wait_until="networkidle")

        # A — Command Centre
        wait_render(page, "Command Centre")
        page.screenshot(path=str(OUT / "01_command_centre.png"), full_page=True)

        # B — Play Forensics
        click_nav(page, "Play Forensics")
        wait_render(page, "Play Forensics")
        page.screenshot(path=str(OUT / "02_play_forensics.png"), full_page=True)

        # C — Counterfactual Lab (run an estimate)
        click_nav(page, "Counterfactual Lab")
        wait_render(page, "Counterfactual Lab")
        try:
            page.get_by_text("Estimate from matched history").click()
            time.sleep(3)
        except Exception:
            pass
        page.screenshot(path=str(OUT / "03_counterfactual_lab.png"), full_page=True)

        # D — Ask PocketShift (ask an example)
        click_nav(page, "Ask PocketShift")
        wait_render(page, "Ask PocketShift")
        try:
            page.get_by_text("Why did this protection fail?").first.click()
            time.sleep(4)
        except Exception:
            pass
        page.screenshot(path=str(OUT / "04_ask_pocketshift.png"), full_page=True)

        # F — Energy & Fatigue (tracking)
        click_nav(page, "Energy & Fatigue")
        wait_render(page, "Energy")
        time.sleep(4)  # tracking tables may build on first load
        page.screenshot(path=str(OUT / "06_energy_fatigue.png"), full_page=True)

        # E — Player Preparation (create a brief)
        click_nav(page, "Player Preparation")
        wait_render(page, "Player Preparation")
        try:
            page.get_by_text("Create Preparation Brief").click()
            time.sleep(3)
        except Exception:
            pass
        page.screenshot(path=str(OUT / "05_player_preparation.png"), full_page=True)

        browser.close()
    print("Screenshots written to", OUT)


if __name__ == "__main__":
    main()
