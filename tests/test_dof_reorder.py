"""`femtools-build-dof-reorder` against the dolfin installed here: the copied function
compiles against its headers and exports exactly the symbol libdolfin exports, and the build
either passes its PLT check or refuses with that reason; a failed compile is refused."""
import shutil

import pytest

pytest.importorskip("dolfin")
pytestmark = [pytest.mark.dolfin]

from femtools import dof_reorder  # noqa: E402


@pytest.mark.skipif(not (shutil.which('c++') and shutil.which('pkg-config')
                         and shutil.which('objdump')), reason='needs c++, pkg-config, objdump')
def test_builds_with_libdolfins_symbol(tmp_path):
    out = str(tmp_path / 'libdof_reorder_star.so')
    try:
        dolfin = dof_reorder.build(out)
    except RuntimeError as e:
        # the one refusal a correct build may meet: this dolfin binds the call inside itself
        assert 'not through its PLT' in str(e), e
        dolfin = str(e).split(': ', 1)[1].split(' calls ')[0]
    ours = dof_reorder._exported(out)
    assert ours is not None and dof_reorder.SYMBOL_PART in ours
    assert ours == dof_reorder._exported(dolfin)


def test_refuses_a_failed_compile(tmp_path, monkeypatch):
    monkeypatch.setattr(dof_reorder, 'SOURCE', str(tmp_path / 'missing.cpp'))
    with pytest.raises(RuntimeError, match='compile failed'):
        dof_reorder.build(str(tmp_path / 'x.so'))
