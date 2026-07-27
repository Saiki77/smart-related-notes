// Maps bare `obsidian` imports onto bench/obsidian-stub.mjs so the real plugin
// sources can be loaded in node. Registered via --import.
import { register } from "node:module";
import { pathToFileURL } from "node:url";
register("./stub-loader.mjs", pathToFileURL("./bench/"));
