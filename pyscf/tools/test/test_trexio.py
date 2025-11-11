import pyscf
from pyscf.tools import trexio
import os
import numpy as np
import tempfile
import pytest

ANGSTROM_TO_BOHR = 1.0 / 0.5291772083
DIFF_TOL = 1e-10

import numpy as np
import types
from pprint import pformat

# 表示を見やすく
np.set_printoptions(precision=6, suppress=True)

def _is_attr(name, obj):
    # 呼び出しで副作用のありそうなものやメソッドは除外
    if name.startswith("__") and name.endswith("__"):
        return False
    try:
        val = getattr(obj, name)
    except Exception:
        return False
    if isinstance(val, (types.FunctionType, types.MethodType)):
        return False
    return True

def _brief(val, maxlen=200):
    """値を短く整形（numpy配列は shape / dtype / 最初の数値だけ）"""
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            head = val.ravel()[:6]
            return f"ndarray(shape={val.shape}, dtype={val.dtype}, head={head})"
    except Exception:
        pass
    s = repr(val)
    if len(s) > maxlen:
        s = s[:maxlen] + "...]trunc"
    return s

def dump_cell(cell, sort_private_last=True):
    names = set(dir(cell)) | set(getattr(cell, "__dict__", {}).keys())
    items = []
    for n in names:
        if not _is_attr(n, cell):
            continue
        try:
            v = getattr(cell, n)
        except Exception as e:
            v = f"<error: {e!r}>"
        items.append((n, v))
    # 並べ替え（先に公開属性、後に _private）
    if sort_private_last:
        items.sort(key=lambda kv: (kv[0].startswith("_"), kv[0]))
    else:
        items.sort(key=lambda kv: kv[0])
    # 文字列化
    lines = []
    for k, v in items:
        lines.append(f"{k}: {_brief(v)}")
    return "\n".join(lines)

def diff_cells(c0, c1, atol=0.0, rtol=0.0, show_equal=False):
    keys = sorted(set(dir(c0)) | set(getattr(c0, "__dict__", {}).keys()) |
                  set(dir(c1)) | set(getattr(c1, "__dict__", {}).keys()))
    diffs = []
    equals = []
    for k in keys:
        if not _is_attr(k, c0) and not _is_attr(k, c1):
            continue
        try:
            v0 = getattr(c0, k)
        except Exception as e:
            v0 = f"<error: {e!r}>"
        try:
            v1 = getattr(c1, k)
        except Exception as e:
            v1 = f"<error: {e!r}>"

        same = False
        # numpy配列は allclose/equal
        try:
            import numpy as np
            if isinstance(v0, np.ndarray) and isinstance(v1, np.ndarray):
                same = (v0.shape == v1.shape) and np.allclose(v0, v1, atol=atol, rtol=rtol, equal_nan=True)
            elif isinstance(v0, (int, float, complex)) and isinstance(v1, (int, float, complex)):
                same = np.isclose(v0, v1, atol=atol, rtol=rtol, equal_nan=True)
            else:
                same = (v0 == v1)
        except Exception:
            same = (repr(v0) == repr(v1))

        if same:
            if show_equal:
                equals.append(f"{k}: {_brief(v0)}")
        else:
            diffs.append(f"[{k}]\n  c0: {_brief(v0)}\n  c1: {_brief(v1)}")

    out = []
    if diffs:
        out.append("=== DIFFS ===")
        out.append("\n".join(diffs))
    if show_equal and equals:
        out.append("\n=== EQUALS ===")
        out.append("\n".join(equals))
    return "\n".join(out) if out else "No differences under given tolerances."

#################################################################
# reading/writing `mol` from/to trexio file
#################################################################

## molecule, segment contraction (6-31g), all-electron
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g**', cart=cart)
        s0 = mol0.intor('int1e_ovlp')
        t0 = mol0.intor('int1e_kin')
        v0 = mol0.intor('int1e_nuc')
        trexio.to_trexio(mol0, filename)
        mol1 = trexio.mol_from_trexio(filename)
        s1 = mol1.intor('int1e_ovlp')
        t1 = mol1.intor('int1e_kin')
        v1 = mol1.intor('int1e_nuc')
        assert abs(s0 - s1).max() < DIFF_TOL
        assert abs(t0 - t1).max() < DIFF_TOL
        assert abs(v0 - v1).max() < DIFF_TOL

## molecule, general contraction (ccpv5z), all-electron
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_ae_ccpv5z(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='C', basis='ccpv5z', cart=cart)
        s0 = mol0.intor('int1e_ovlp')
        t0 = mol0.intor('int1e_kin')
        v0 = mol0.intor('int1e_nuc')
        trexio.to_trexio(mol0, filename)
        mol1 = trexio.mol_from_trexio(filename)
        s1 = mol1.intor('int1e_ovlp')
        t1 = mol1.intor('int1e_kin')
        v1 = mol1.intor('int1e_nuc')
        assert abs(s0 - s1).max() < DIFF_TOL
        assert abs(t0 - t1).max() < DIFF_TOL
        assert abs(v0 - v1).max() < DIFF_TOL

## molecule, general contraction (ano), all-electron
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_ae_ano(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='C', basis='ano', cart=cart)
        s0 = mol0.intor('int1e_ovlp')
        t0 = mol0.intor('int1e_kin')
        v0 = mol0.intor('int1e_nuc')
        trexio.to_trexio(mol0, filename)
        mol1 = trexio.mol_from_trexio(filename)
        s1 = mol1.intor('int1e_ovlp')
        t1 = mol1.intor('int1e_kin')
        v1 = mol1.intor('int1e_nuc')
        assert abs(s0 - s1).max() < DIFF_TOL
        assert abs(t0 - t1).max() < DIFF_TOL
        assert abs(v0 - v1).max() < DIFF_TOL

## molecule, segment contraction (ccecp-cc-pVQZ), ccecp
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_ccecp_ccecp_ccpvqz(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccecp-ccpvqz', ecp='ccecp', cart=cart)
        s0 = mol0.intor('int1e_ovlp')
        t0 = mol0.intor('int1e_kin')
        v0 = mol0.intor('int1e_nuc')
        trexio.to_trexio(mol0, filename)
        mol1 = trexio.mol_from_trexio(filename)
        s1 = mol1.intor('int1e_ovlp')
        t1 = mol1.intor('int1e_kin')
        v1 = mol1.intor('int1e_nuc')
        assert abs(s0 - s1).max() < DIFF_TOL
        assert abs(t0 - t1).max() < DIFF_TOL
        assert abs(v0 - v1).max() < DIFF_TOL

## PBC, k=gamma, segment contraction (6-31g), all-electron
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_gamma_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kpt = np.zeros(3)
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g**', a=np.diag([3.0, 3.0, 5.0])*ANGSTROM_TO_BOHR, unit="Bohr")
        print("cell0.nuc_mod:", getattr(cell0, "nuc_mod", "ewald"))
        print("cell0.dimension:", cell0.dimension)
        print("cell0.unit:", getattr(cell0, "unit", "Angstrom"))
        print("cell0.a:\n", cell0.a)
        R_bohr = cell0.atom_coords(unit="Bohr")
        print("cell0.R_bohr:\n", R_bohr)
        print("kpt:\n", kpt)
        s0 = cell0.pbc_intor('int1e_ovlp', kpts=kpt)
        t0 = cell0.pbc_intor('int1e_kin', kpts=kpt)
        v0 = cell0.pbc_intor('int1e_nuc', kpts=kpt)
        trexio.to_trexio(cell0, filename)
        cell1 = trexio.mol_from_trexio(filename)
        cell1.build()
        print("cell1.nuc_mod:", getattr(cell1, "nuc_mod", "ewald"))
        print("cell1.dimension:", cell1.dimension)
        print("cell1.unit:", getattr(cell1, "unit", "Angstrom"))
        print("cell1.a:\n", cell1.a)
        R_bohr = cell1.atom_coords(unit="Bohr")
        print("cell1.R_bohr:\n", R_bohr)
        print("kpt:\n", kpt)
        s1 = cell1.pbc_intor('int1e_ovlp', kpts=kpt)
        t1 = cell1.pbc_intor('int1e_kin', kpts=kpt)
        v1 = cell1.pbc_intor('int1e_nuc', kpts=kpt)

        print(dump_cell(cell0))
        print(dump_cell(cell1))
        print(diff_cells(cell0, cell1, atol=0.0, rtol=0.0, show_equal=False))

        print("nuc_mod:", getattr(cell0, "nuc_mod", None), getattr(cell1, "nuc_mod", None))
        print("precision:", cell0.precision, cell1.precision)
        print("ew_eta:", getattr(cell0, "ew_eta", None), getattr(cell1, "ew_eta", None))
        print("rcut:", getattr(cell0, "rcut", None), getattr(cell1, "rcut", None))
        print("gs:", getattr(cell0, "gs", None), getattr(cell1, "gs", None))
        print("mesh:", getattr(cell0, "mesh", None), getattr(cell1, "mesh", None))
        print("cart:", cell0.cart, cell1.cart)
        np.testing.assert_array_equal(cell0.atom_charges(), cell1.atom_charges())
        nbas = cell1.nbas
        bas_centers0 = np.array([cell0.bas_coord(i) for i in range(nbas)])
        nbas = cell1.nbas
        bas_centers1 = np.array([cell1.bas_coord(i) for i in range(nbas)])
        np.testing.assert_allclose(bas_centers0, bas_centers1, rtol=0, atol=0)

        assert abs(s0 - s1).max() < DIFF_TOL
        assert abs(t0 - t1).max() < DIFF_TOL
        assert abs(v0 - v1).max() < DIFF_TOL

## PBC, k=grid, segment contraction (6-31g), all-electron
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_grid_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 2)
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g**', a=np.diag([3.0, 3.0, 5.0])*ANGSTROM_TO_BOHR, unit="Bohr")
        print("cell0.nuc_mod:", getattr(cell0, "nuc_mod", "ewald"))
        print("cell0.dimension:", cell0.dimension)
        print("cell0.unit:", getattr(cell0, "unit", "Angstrom"))
        print("cell0.a:\n", cell0.a)
        R_bohr = cell0.atom_coords(unit="Bohr")
        print("cell0.R_bohr:\n", R_bohr)
        kpts0 = cell0.make_kpts(kmesh)
        print("kpt:\n", kpts0)
        s0 = np.asarray(cell0.pbc_intor('int1e_ovlp', kpts=kpts0))
        t0 = np.asarray(cell0.pbc_intor('int1e_kin', kpts=kpts0))
        v0 = np.asarray(cell0.pbc_intor('int1e_nuc', kpts=kpts0))
        trexio.to_trexio(cell0, filename)
        cell1 = trexio.mol_from_trexio(filename)
        cell1.build()
        print("cell1.nuc_mod:", getattr(cell1, "nuc_mod", "ewald"))
        print("cell1.dimension:", cell1.dimension)
        print("cell1.unit:", getattr(cell1, "unit", "Angstrom"))
        print("cell1.a:\n", cell1.a)
        R_bohr = cell1.atom_coords(unit="Bohr")
        print("cell1.R_bohr:\n", R_bohr)
        kpts1 = cell1.make_kpts(kmesh)
        print("kpt:\n", kpts1)
        s1 = np.asarray(cell1.pbc_intor('int1e_ovlp', kpts=kpts1))
        t1 = np.asarray(cell1.pbc_intor('int1e_kin', kpts=kpts1))
        v1 = np.asarray(cell1.pbc_intor('int1e_nuc', kpts=kpts1))
        assert abs(s0 - s1).max() < DIFF_TOL
        assert abs(t0 - t1).max() < DIFF_TOL
        assert abs(v0 - v1).max() < DIFF_TOL

#################################################################
# reading/writing `mf` from/to trexio file
#################################################################

## molecule, segment contraction (6-31g), all-electron, RHF
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g*', cart=cart)
        mf0 = mol0.RHF().run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(mf1.mo_coeff - mf0.mo_coeff).max() < DIFF_TOL

## molecule, segment contraction (6-31g), all-electron, UHF
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; H 0 0 1', basis='6-31g*', spin=2, cart=cart)
        mf0 = mol0.UHF().run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(mf1.mo_coeff - mf0.mo_coeff).max() < DIFF_TOL

## molecule, segment contraction (ccecp-cc-pVQZ), ccecp, RHF
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_rhf_ccecp_ccpvqz(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccecp-ccpvqz', ecp='ccecp', cart=cart)
        mf0 = mol0.RHF().run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(mf1.mo_coeff - mf0.mo_coeff).max() < DIFF_TOL

## PBC, k=gamma, segment contraction (6-31g), all-electron, RHF
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_gamma_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        mf0 = pyscf.pbc.scf.RKS(cell0)
        mf0.xc = 'LDA'
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(mf1.mo_coeff - mf0.mo_coeff).max() < DIFF_TOL

## PBC, k=general, segment contraction (6-31g), all-electron, RHF
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_general_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        kfrac = (0.25, 0.25, 0.25)
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        kpt0 = cell0.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf0 = pyscf.pbc.scf.RKS(cell0, kpt=kpt0)
        mf0.xc = 'LDA'
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(mf1.mo_coeff - mf0.mo_coeff).max() < DIFF_TOL

## PBC, k=grid, segment contraction (6-31g), all-electron, RHF
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_grid_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 2)
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        kpts0 = cell0.make_kpts(kmesh)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.KRKS(cell0, kpts=kpts0)
        mf0.xc = 'LDA'
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(np.asarray(mf1.mo_coeff) - np.asarray(mf0.mo_coeff)).max() < DIFF_TOL

## PBC, k=gamma, segment contraction (6-31g), all-electron, UHF
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_gamma_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        mf0 = pyscf.pbc.scf.UKS(cell0)
        mf0.xc = 'LDA'
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(mf1.mo_coeff - mf0.mo_coeff).max() < DIFF_TOL

## PBC, k=general, segment contraction (6-31g), all-electron, UHF
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_general_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        kfrac = (0.25, 0.25, 0.25)
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        kpt0 = cell0.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf0 = pyscf.pbc.scf.UKS(cell0, kpt=kpt0)
        mf0.xc = 'LDA'
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(mf1.mo_coeff - mf0.mo_coeff).max() < DIFF_TOL

## PBC, k=grid, segment contraction (6-31g), all-electron, UHF
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_grid_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 2)
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        kpts0 = cell0.make_kpts(kmesh)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.KUKS(cell0, kpts=kpts0)
        mf0.xc = 'LDA'
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(np.asarray(mf1.mo_coeff) - np.asarray(mf0.mo_coeff)).max() < DIFF_TOL

#################################################################
# reading/writing `mol` from/to trexio file + SCF run.
#################################################################

## molecule, segment contraction (6-31g), all-electron, RHF
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_scf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g*', cart=cart)
        trexio.to_trexio(mol0, filename)
        mf0 = mol0.RHF().run()
        e0 = mf0.e_tot
        mol1 = trexio.mol_from_trexio(filename)
        mf1 = mol1.RHF().run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

## PBC, k=gamma, segment contraction (6-31g), all-electron, RKS
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_gamma_scf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.RKS(cell0)
        mf0.xc = 'LDA'
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        mf1 = pyscf.pbc.scf.RKS(cell1)
        mf1.xc = 'LDA'
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

## PBC, k=gamma, segment contraction (6-31g), all-electron, UKS
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_gamma_scf_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.UKS(cell0)
        mf0.xc = 'LDA'
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        mf1 = pyscf.pbc.scf.UKS(cell1)
        mf1.xc = 'LDA'
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL


## PBC, k=general, segment contraction (6-31g), all-electron, RKS
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_general_scf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        kfrac = (0.25, 0.25, 0.25)
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        trexio.to_trexio(cell0, filename)
        kpt0 = cell0.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf0 = pyscf.pbc.scf.RKS(cell0, kpt=kpt0)
        mf0.xc = 'LDA'
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        kpt1 = cell1.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf1 = pyscf.pbc.scf.RKS(cell1, kpt=kpt1)
        mf1.xc = 'LDA'
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

## PBC, k=general, segment contraction (6-31g), all-electron, UKS
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_general_scf_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        kfrac = (0.25, 0.25, 0.25)
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        trexio.to_trexio(cell0, filename)
        kpt0 = cell0.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf0 = pyscf.pbc.scf.UKS(cell0, kpt=kpt0)
        mf0.xc = 'LDA'
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        kpt1 = cell1.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf1 = pyscf.pbc.scf.UKS(cell1, kpt=kpt1)
        mf1.xc = 'LDA'
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

## PBC, k=grid, segment contraction (6-31g), all-electron, RKS
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_grid_scf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 2)
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        kpts0 = cell0.make_kpts(kmesh)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.KRKS(cell0, kpts=kpts0)
        mf0.xc = 'LDA'
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        kpts1 = cell1.make_kpts(kmesh)
        mf1 = pyscf.pbc.scf.KRKS(cell1, kpts=kpts1)
        mf1.xc = 'LDA'
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

## molecule, segment contraction (6-31g), all-electron, UHF
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_scf_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; H 0 0 1', basis='6-31g*', spin=2, cart=cart)
        trexio.to_trexio(mol0, filename)
        mf0 = mol0.UHF().run()
        e0 = mf0.e_tot
        mol1 = trexio.mol_from_trexio(filename)
        mf1 = mol1.UHF().run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

## molecule, segment contraction (ccecp-cc-pVQZ), ccecp, RHF
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_rhf_ccecp_ccpvqz(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccecp-ccpvqz', ecp='ccecp', cart=cart)
        trexio.to_trexio(mol0, filename)
        mf0 = mol0.RHF().run()
        e0 = mf0.e_tot
        mol1 = trexio.mol_from_trexio(filename)
        mf1 = mol1.RHF().run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='F 0 0 0; F 0 0 1', basis='ccecp-ccpvqz', ecp='ccecp', cart=cart)
        trexio.to_trexio(mol0, filename)
        mf0 = mol0.RHF().run()
        e0 = mf0.e_tot
        mol1 = trexio.mol_from_trexio(filename)
        mf1 = mol1.RHF().run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; H 0 0 1', basis='ccecp-ccpvqz', ecp='ccecp', cart=cart)
        trexio.to_trexio(mol0, filename)
        mf0 = mol0.RHF().run()
        e0 = mf0.e_tot
        mol1 = trexio.mol_from_trexio(filename)
        mf1 = mol1.RHF().run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

## molecule, segment contraction (ccecp-cc-pVQZ), ccecp, UHF
@pytest.mark.skip(reason='debug')
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_rhf_ccecp_ccpvqz(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccecp-ccpvqz', ecp='ccecp', spin=2, cart=cart)
        trexio.to_trexio(mol0, filename)
        mf0 = mol0.UHF().run()
        e0 = mf0.e_tot
        mol1 = trexio.mol_from_trexio(filename)
        mf1 = mol1.UHF().run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL
