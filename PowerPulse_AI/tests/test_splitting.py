from ml.dataset import load_clean_dataset
from ml.splitting import ambient_split_summary, partition_frame, split_batteries


def test_battery_split_is_disjoint_reproducible_and_temperature_balanced():
    frame = load_clean_dataset()
    train_ids, test_ids = split_batteries(frame)
    again_train, again_test = split_batteries(frame)
    assert train_ids == again_train
    assert test_ids == again_test
    assert len(test_ids) == 8
    assert len(train_ids) == 24
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(frame["battery_id"])

    train, test = partition_frame(frame, train_ids, test_ids)
    assert set(train["battery_id"]).isdisjoint(test["battery_id"])
    assert len(train) + len(test) == len(frame)

    summary = ambient_split_summary(frame, train_ids, test_ids)
    for row in summary:
        assert row["test_batteries"], row
        assert row["train_batteries"], row
        assert set(row["train_batteries"]).isdisjoint(row["test_batteries"])
