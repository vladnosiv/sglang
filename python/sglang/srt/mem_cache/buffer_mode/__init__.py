"""Buffer-only HiCache transient storage pipelines.

The default backend uses operation-owned Host bounces. The optional
``gpu_transient`` backend uses fixed registered CUDA TX/RX rings. Both remain
non-persistent staging tiers and publish restored pages only through
``BufferModePipeline`` on the scheduler thread.
"""
