import os
from tempfile import TemporaryDirectory
import pytest
import numpy


@pytest.fixture(autouse=True)
def doctest_setup_teardown(request):
    doctest_plugin = request.config.pluginmanager.getplugin("doctest")
    if isinstance(request.node, doctest_plugin.DoctestItem):
        tmp_dir = TemporaryDirectory()
        org_dir = os.getcwd()
        os.chdir(tmp_dir.name)
        yield
        os.chdir(org_dir)
        tmp_dir.cleanup()
    else:
        yield
    import matplotlib.pyplot as plt
    plt.close('all')


@pytest.fixture(autouse=True)
def add_np_am(doctest_namespace):
        import atomap.api as am
        doctest_namespace['np'] = numpy
        doctest_namespace['am'] = am
