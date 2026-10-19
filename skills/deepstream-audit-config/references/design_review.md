# Design review hotspots

Start with the public DeepStream documentation when checking supported plugin or
configuration semantics:

- [Plugins reference](https://nvidia-deepstream.docs.buildwithfern.com/deepstream/9.2/text/DS_plugin_Intro)
- [Gst-nvinfer](https://nvidia-deepstream.docs.buildwithfern.com/deepstream/9.2/text/DS_plugin_gst-nvinfer)
- [Gst-nvinferserver](https://nvidia-deepstream.docs.buildwithfern.com/deepstream/9.2/text/DS_plugin_gst-nvinferserver)
- [Troubleshooting](https://nvidia-deepstream.docs.buildwithfern.com/deepstream/9.2/text/DS_troubleshooting)

For contributor reviews in a source checkout, inspect the implementation paths
that apply:

- `src/apps/common/src/deepstream_config_file_parser.c`
- `src/apps/common/src/deepstream_streammux.c`
- `src/apps/common/src/deepstream_source_bin.c`
- `src/apps/sample_apps/deepstream-test5/deepstream_test5_app_main.c`
- `src/gst-plugins/gst-nvmultiurisrcbin/gstdsnvmultiurisrcbin.cpp`
- `src/gst-utils/gstnvdscustomhelper/gst-nvmultiurisrcbincreator.cpp`

Check parser semantics **and** runtime plugin properties — keys can parse yet still
be wrong for the use case.

If the source checkout is unavailable, use the public documentation plus
[rules.md](rules.md); do not invent parser behavior.
