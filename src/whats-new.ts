import { App, Modal } from "obsidian";

// One-time "what's new" shown after an update, in the spirit of Excalidraw's
// release modal. WHATS_NEW_ID names the FEATURE release the bundled highlights
// describe; it only changes when there is something worth a popup, so patch
// releases update silently and nobody sees the same highlights twice.
export const WHATS_NEW_ID = "4.0";

export class WhatsNewModal extends Modal {
  constructor(
    app: App,
    private openReaderSettings: () => void,
  ) {
    super(app);
  }

  onOpen(): void {
    this.modalEl.addClass("rn-whatsnew");
    this.titleEl.setText("Smart Related Notes 4.0");
    const c = this.contentEl;

    const section = (title: string, body: string): HTMLElement => {
      const d = c.createDiv({ cls: "rn-whatsnew-section" });
      d.createEl("h3", { text: title });
      d.createEl("p", { text: body });
      return d;
    };

    const reader = section(
      "A reader, working in the background",
      "A small local text model can now read your notes while Obsidian is idle. " +
        "Every card gets a one-line summary (hover for a fuller one), and the active " +
        "note gets tag suggestions you add with one click. It is off by default, runs " +
        "entirely on your machine, and stores its files outside the vault so sync " +
        "never sees them.",
    );
    const cta = reader.createEl("button", { text: "Turn it on in settings", cls: "mod-cta" });
    cta.addEventListener("click", () => {
      this.close();
      this.openReaderSettings();
    });

    section(
      "Works on locked-down networks",
      "If your network blocks downloads, run \"Reader: offline setup\" from the " +
        "command palette. Your browser fetches the files, and the plugin imports and " +
        "verifies each one automatically as the download finishes.",
    );

    section(
      "Pin and peek",
      "The pin in the panel header freezes the ranking to one note, so opening " +
        "results keeps the list that surfaced them. Cards now open in a new tab with " +
        "cmd/ctrl click or middle click, and a right-click menu offers new tab, " +
        "split, or here.",
    );

    const foot = c.createDiv({ cls: "rn-whatsnew-foot" });
    foot.appendText("The surprising-connections section of the insights report was removed. ");
    foot.createEl("a", {
      text: "Full release notes",
      href: "https://github.com/Saiki77/smart-related-notes/releases",
    });
  }

  onClose(): void {
    this.contentEl.empty();
  }
}
