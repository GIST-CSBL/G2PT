"""Synthetic, CPU-only constructor tests without Torch/sgkit installation.

The actual dataset classes are compiled from source, including inheritance and
__getitem__. Only tensor primitives, the tree parser and the PLINK reader are
substituted. Real pandas file loading/alignment and NumPy operations are used.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pandas as pd
import pytest


class Tensor(np.ndarray):
    def contiguous(self):
        return self


def tensor(value, dtype=None):
    return np.asarray(value, dtype=dtype).view(Tensor)


TORCH = SimpleNamespace(
    long=np.int64, int=np.int32, float32=np.float32, tensor=tensor, as_tensor=tensor,
    arange=lambda n, dtype=None: tensor(np.arange(n), dtype),
    full=lambda shape, value, dtype=None: tensor(np.full(shape, value), dtype),
    cat=lambda values, dim=0: tensor(np.concatenate(values, axis=dim)),
    clip=lambda value, min, max: tensor(np.clip(value, min, max)),
)


class Array:
    def __init__(self, values):
        self.values = np.asarray(values)

    def as_numpy(self):
        return self.values


class PlinkData:
    def __init__(self, phenotype=(2., 1., 2.)):
        self.sample_id = Array(["p3", "p1", "p2"])
        self.sample_family_id = Array(["f3", "f1", "f2"])
        self.sample_sex = Array([0, 1, 0])
        self.sample_phenotype = Array(phenotype)
        self.variant_id = Array([101, 102])
        self.contig_id = Array(["1"])
        self.variant_position = Array([10, 20])
        self.variant_contig = Array([0, 0])
        self.call_genotype = Array([[[0, 0], [0, 1], [1, 1]], [[0, 1], [1, 1], [0, 0]]])

    def __getitem__(self, key):
        return getattr(self, key)


@pytest.fixture
def datasets():
    path = Path(__file__).resolve().parents[1] / "src/utils/data/dataset/genotype_datasets.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"GenotypeDataset", "TSVDataset", "PLINKDataset"}]
    reader = SimpleNamespace(read_plink=lambda path: PlinkData())
    namespace = {"np": np, "pd": pd, "torch": TORCH, "Dataset": object,
                 "SNPTreeParser": object, "plink": reader, "time": time}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in ["GenotypeDataset", "TSVDataset", "PLINKDataset"]}, reader=reader)


@pytest.fixture
def inputs(tmp_path):
    cov = pd.DataFrame({"FID": ["f2", "f3", "f1"], "IID": ["p2", "p3", "p1"],
                        "SEX": [0, 0, 1], "PHENOTYPE": [2.5, np.nan, 1.25]})
    cov.to_csv(tmp_path / "cov.tsv", sep="\t", index=False)
    pd.DataFrame({101: [0, 1, 2], 102: [2, 0, 1]}, index=["p3", "p1", "p2"]).to_csv(tmp_path / "geno.tsv", sep="\t")
    parser = SimpleNamespace(n_snps=2, n_genes=1, n_systems=1, snp2ind={101: 0, 102: 1}, block=False,
                             set_snp2chr=lambda values: None, set_snp2pos=lambda values: None)
    return tmp_path, cov, parser


def assert_metadata(dataset, ids, bt=(), qt=None):
    assert list(dataset.pheno_ids) == ids
    assert dataset.pheno2ind == {name: i for i, name in enumerate(ids)}
    assert dataset.ind2pheno == dict(enumerate(ids))
    assert dataset.n_pheno == len(ids) and dataset.pheno_range == list(range(len(ids)))
    quantitative = ids if qt is None else list(qt)
    assert list(dataset.qt) == quantitative and list(dataset.bt) == list(bt)
    assert dataset.qt_inds == [ids.index(name) for name in quantitative]
    assert dataset.bt_inds == [ids.index(name) for name in bt]
    assert dataset.pheno2type == {**{name: "qt" for name in quantitative}, **{name: "bt" for name in bt}}
    assert dataset.has_phenotype is True


def assert_genotype_alignment(dataset, kind):
    # Expected first two SNP indices, computed independently from each synthetic
    # participant's allele counts with the existing default allele flip.
    expected = ({"p3": [4, 1], "p1": [2, 5], "p2": [0, 3]} if kind == "TSVDataset"
                else {"p3": [4, 3], "p1": [2, 1], "p2": [0, 5]})
    for i, iid in dataset.ind2iid.items():
        np.testing.assert_array_equal(dataset[i]["genotype"]["snp"][:2], expected[iid])


@pytest.mark.parametrize("kind", ["GenotypeDataset", "TSVDataset", "PLINKDataset"])
@pytest.mark.parametrize("binary", [False, True])
def test_covariate_fallback_metadata_and_order(datasets, inputs, kind, binary):
    path, cov, parser = inputs
    if binary:
        cov["PHENOTYPE"] = [1., np.nan, 0.]
        cov.to_csv(path / "cov.tsv", sep="\t", index=False)
    args = [parser] + ([str(path / "geno.tsv")] if kind != "GenotypeDataset" else [])
    kwargs = {"bt": ["PHENOTYPE"], "pheno_ids": ["PHENOTYPE"]} if binary else {}
    dataset = getattr(datasets, kind)(*args, cov=str(path / "cov.tsv"), **kwargs)
    assert_metadata(dataset, ["PHENOTYPE"], bt=["PHENOTYPE"] if binary else (), qt=[] if binary else None)
    expected = dict(zip(cov.IID, cov.PHENOTYPE.fillna(-9)))
    for i, row in dataset.cov_df.iterrows():
        assert dataset.pheno_df.iloc[i].IID == row.IID
        assert dataset.pheno_df.iloc[i].FID == row.FID
        assert dataset.pheno_df.iloc[i].PHENOTYPE == expected[row.IID]
        np.testing.assert_array_equal(dataset[i]["phenotype"], [expected[row.IID]])
        np.testing.assert_array_equal(dataset[i]["phenotype_indices"], [0])
    if kind == "TSVDataset":
        assert dataset.cov_df.IID.tolist() == ["p3", "p1", "p2"]
    if kind != "GenotypeDataset":
        assert list(dataset.ind2iid.values()) == dataset.cov_df.IID.tolist()
        assert_genotype_alignment(dataset, kind)
    assert pd.read_csv(path / "cov.tsv", sep="\t").PHENOTYPE.isna().sum() == 1


@pytest.mark.parametrize("kind", ["GenotypeDataset", "TSVDataset", "PLINKDataset"])
@pytest.mark.parametrize("explicit_ids", [False, True])
def test_separate_file_preserved(datasets, inputs, kind, explicit_ids):
    path, cov, parser = inputs
    pheno = cov[["FID", "IID"]].copy()
    pheno["BINARY"] = [1., 0., np.nan]
    pheno["QUANT"] = [1.5, np.nan, -9.]
    # Deliberately differ from covariates for derived datasets, which reorder by IID.
    if kind != "GenotypeDataset":
        pheno = pheno.iloc[[2, 0, 1]]
    pheno.to_csv(path / "pheno.tsv", sep="\t", index=False)
    ids = ["QUANT", "BINARY"] if explicit_ids else ["BINARY", "QUANT"]
    args = [parser] + ([str(path / "geno.tsv")] if kind != "GenotypeDataset" else [])
    dataset = getattr(datasets, kind)(*args, cov=str(path / "cov.tsv"), pheno=str(path / "pheno.tsv"),
                                    pheno_ids=ids if explicit_ids else (), bt=["BINARY"], qt=["QUANT"])
    assert_metadata(dataset, ids, bt=["BINARY"], qt=["QUANT"])
    by_id = pheno.set_index("IID").fillna(-9)
    for i, row in dataset.cov_df.iterrows():
        assert dataset.pheno_df.iloc[i].IID == row.IID
        np.testing.assert_array_equal(dataset[i]["phenotype"], by_id.loc[row.IID, ids].astype(float).values)
    if kind != "GenotypeDataset":
        assert_genotype_alignment(dataset, kind)


@pytest.mark.parametrize("kind", ["GenotypeDataset", "TSVDataset"])
def test_missing_phenotypes_give_actionable_error(datasets, inputs, kind):
    path, cov, parser = inputs
    cov.drop(columns="PHENOTYPE").to_csv(path / "cov.tsv", sep="\t", index=False)
    args = [parser] + ([str(path / "geno.tsv")] if kind == "TSVDataset" else [])
    with pytest.raises(ValueError, match="[Pp]henotype"):
        getattr(datasets, kind)(*args, cov=str(path / "cov.tsv"))


@pytest.mark.parametrize("external_cov", [False, True])
@pytest.mark.parametrize("binary", [False, True])
def test_plink_late_fam_initialization(datasets, inputs, external_cov, binary):
    path, cov, parser = inputs
    cov.drop(columns="PHENOTYPE").to_csv(path / "cov.tsv", sep="\t", index=False)
    kwargs = {"bt": ["PHENOTYPE"]} if binary else {}
    dataset = datasets.PLINKDataset(parser, "synthetic", cov=str(path / "cov.tsv") if external_cov else None, **kwargs)
    assert_metadata(dataset, ["PHENOTYPE"], bt=["PHENOTYPE"] if binary else (), qt=[] if binary else None)
    expected = {"p3": 1., "p1": 0., "p2": 1.}  # existing .fam coding: subtract one
    assert list(dataset.ind2iid.values()) == dataset.cov_df.IID.tolist() == dataset.pheno_df.IID.tolist()
    for i, iid in dataset.ind2iid.items():
        np.testing.assert_array_equal(dataset[i]["phenotype"], [expected[iid]])
    assert_genotype_alignment(dataset, "PLINKDataset")
    if not external_cov:
        assert dataset.cov_ids == ["SEX"] and dataset.n_cov == 2
        assert dataset.cov_mean_dict == {} and dataset.cov_std_dict == {}


def test_plink_missing_fam_values_keep_alignment(datasets, inputs):
    path, cov, parser = inputs
    datasets.reader.read_plink = lambda path: PlinkData((2., -1., np.nan))
    dataset = datasets.PLINKDataset(parser, "synthetic", bt=["PHENOTYPE"])
    assert_metadata(dataset, ["PHENOTYPE"], bt=["PHENOTYPE"], qt=[])
    assert dataset.cov_df.IID.tolist() == dataset.pheno_df.IID.tolist() == ["p3", "p2"]
    assert list(dataset.ind2iid.values()) == ["p3", "p2"]
    assert dataset.pheno_df.PHENOTYPE.tolist() == [1., -9.]
    assert_genotype_alignment(dataset, "PLINKDataset")


@pytest.mark.parametrize("kind", ["GenotypeDataset", "TSVDataset", "PLINKDataset"])
def test_missing_requested_phenotype_is_informative(datasets, inputs, kind):
    path, cov, parser = inputs
    args = [parser] + ([str(path / "geno.tsv")] if kind != "GenotypeDataset" else [])
    with pytest.raises(ValueError, match="UNKNOWN"):
        getattr(datasets, kind)(*args, cov=str(path / "cov.tsv"), pheno_ids=["UNKNOWN"])


def test_separate_file_with_fam_covariates_has_precedence(datasets, inputs):
    path, cov, parser = inputs
    pheno = cov[["FID", "IID", "PHENOTYPE"]].copy()
    pheno.to_csv(path / "pheno.tsv", sep="\t", index=False)
    datasets.reader.read_plink = lambda path: PlinkData((-1., -1., -1.))
    dataset = datasets.PLINKDataset(parser, "synthetic", pheno=str(path / "pheno.tsv"))
    assert_metadata(dataset, ["PHENOTYPE"])
    assert len(dataset) == 3
    expected = pheno.set_index("IID").PHENOTYPE.fillna(-9).to_dict()
    for i, iid in dataset.ind2iid.items():
        assert dataset.pheno_df.iloc[i].IID == iid
        np.testing.assert_array_equal(dataset[i]["phenotype"], [expected[iid]])
    assert_genotype_alignment(dataset, "PLINKDataset")


def test_unknown_phenotype_type_is_informative(datasets, inputs):
    path, cov, parser = inputs
    cov[["FID", "IID", "PHENOTYPE"]].to_csv(path / "pheno.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="UNKNOWN"):
        datasets.GenotypeDataset(parser, str(path / "cov.tsv"), pheno=str(path / "pheno.tsv"), qt=["UNKNOWN"])


@pytest.mark.parametrize("external_cov", [False, True])
def test_missing_fam_phenotype_is_informative(datasets, inputs, external_cov):
    path, cov, parser = inputs
    cov.drop(columns="PHENOTYPE").to_csv(path / "cov.tsv", sep="\t", index=False)
    data = PlinkData()
    del data.sample_phenotype
    datasets.reader.read_plink = lambda path: data
    with pytest.raises(ValueError, match="[Pp]henotype"):
        datasets.PLINKDataset(parser, "synthetic", cov=str(path / "cov.tsv") if external_cov else None)
