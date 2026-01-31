+++
title = "Welcome to My Blog"
author = ["Preston Davis"]
description = "First post on my new Hugo-powered blog"
date = 2026-01-31
tags = ["meta", "hugo"]
draft = false
toc = true
+++

Welcome to my new blog! This site is built with:

-   [Hugo](https://gohugo.io/) - A fast static site generator
-   [ox-hugo](https://ox-hugo.scripter.co/) - Org mode to Hugo exporter for Emacs
-   [Doom Emacs](https://www.gnu.org/software/emacs/) - My preferred development environment


## Why Hugo? {#why-hugo}

I previously used Ghost for blogging, but I wanted something that:

1.  Integrates with my Org-mode workflow in Emacs
2.  Allows writing in plain text (version controlled)
3.  Deploys as a static site (fast and secure)
4.  Doesn't require a database

Hugo checks all these boxes. Combined with ox-hugo, I can write posts in Org mode
and export them directly to Hugo-compatible Markdown.


## The Setup {#the-setup}

Posts are written in a single `posts.org` file using Org subtrees. Each subtree
becomes a separate blog post when exported. The workflow is:

1.  Write post as Org subtree
2.  Export with `C-c C-e H H` (or `SPC o B e`)
3.  Commit and push to GitHub
4.  Webhook triggers rebuild on server
5.  Post is live!

Stay tuned for more posts on Emacs, DevOps, and other technical topics.
