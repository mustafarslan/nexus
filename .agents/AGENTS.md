# Nexus Development & Evaluation Rules

## 1. Test Harness and Evaluation Integrity
- **Isolate Routing from Extraction**: When evaluating argument generation/extraction accuracy, always isolate routing errors from argument mismatches. Only evaluate argument accuracy on cases where the routed tool matches the expected tool, or force the correct tool sequence.
- **Robust Value Normalization**: Always compare string parameters using normalized matching (e.g., stripping spaces, hyphens, underscores, and converting to lowercase) to avoid false negatives on minor formatting conventions.
- **Flexible List Support**: Support matching list/tuple options for expected values where multiple semantic interpretations are equally valid.

## 2. Argument Generation Guidance
- **Missing Required Identifiers**: When a parameter is required by the tool schema but absent from the user query, provide a clear, concise fallback rule in the generation prompt (e.g., defaulting the `owner` field to `"user"`) to prevent the model from mis-slotting other query parameters into that required field.
- **Sanitize GitHub Identifiers**: Repository and owner names on GitHub cannot contain spaces. Instruct the model to convert spaced names from user queries into kebab-case (hyphenated) or snake-case (underscored) format when mapping them to repository/owner fields.
