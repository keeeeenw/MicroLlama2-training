# Privacy and publication scope

This is a curated training-code release with a fresh Git history. Private operational notes, raw receipts, account/run identities, machine/IP/SSH details, provider-control code, local recovery packages, credentials and unpublished Mac continuation work are intentionally excluded. Public model/data identifiers, source hashes, published author name and model citations are retained.

Keep the original operational repository private. Deleting a file in a newer commit does not remove it from that repository's earlier commits.

Before making this repository public, review the staged tree and history and run a secret scanner. The release review uses Gitleaks plus a separate check for personal paths and infrastructure identifiers; no scanner can guarantee the absence of every kind of private information. Hashes/revisions are not credentials and must not be removed merely because a generic secret rule flags them.

Use environment variables or your normal CLI login for credentials. Never commit `.env`, authentication caches, logs, checkpoints, copied operational records or access keys. If an actual credential is ever published, revoke/rotate it and follow the hosting provider's removal process.
