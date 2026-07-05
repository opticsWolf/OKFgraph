---
title: Chunking Guide
path: chunking-guide.md
tags: [chunking, retrieval, text-processing]
---

# Chunking Guide

Text chunking splits long documents into manageable segments for embedding and retrieval.

## Why Chunk?

Documents can be hundreds or thousands of words long. Embedding the entire document loses fine-grained detail. Chunking preserves section-level semantics.

## Chunk Size and Overlap

The default chunk size is 512 words with 64 words of overlap between adjacent chunks. This overlap ensures that context spanning chunk boundaries is not lost.

### Choosing Chunk Size

- **Small chunks** (100-200 words): More granular, better for precise retrieval
- **Medium chunks** (300-500 words): Balanced approach for most use cases
- **Large chunks** (600+ words): More context per chunk, fewer total chunks

### Overlap Strategy

Overlap is computed by extracting the tail of the previous chunk and prefixing it to the next chunk. The overlap payload tracks which words belong to the overlap region.

## Reconstruction

After retrieval, chunks can be reassembled into the original document. Block type markers (heading, paragraph, list) determine the correct delimiters between chunks.

## See Also

- [[intro]] for knowledge graph fundamentals
