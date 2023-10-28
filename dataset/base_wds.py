import webdataset as wds

def base_wds(
        urls, map_key, map_tran, seed=None,
        length=10000, end_epoch=0, epoch=0, bufsize=1000, debug=False,
        rc_batchsampler=None,
):
    shards = wds.SimpleShardList(urls, seed=seed)
    shards.epoch = epoch
    if debug:
        map_key.extend(["__url__", "__key__"])

    to_tuple = wds.to_tuple(*map_key)
    if isinstance(map_tran, list):
        if debug:
            map_tran.extend([lambda x: x, lambda x: x])
        map_tuple = wds.map_tuple(*map_tran)
    else:
        map_tuple = wds.map(map_tran)
    pipeline = [
        shards,
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(),
    ]
    if seed:
        pipeline.append(
            wds.detshuffle(
                bufsize=bufsize, initial=bufsize,
                seed=seed, epoch=epoch
            )
        )
    pipeline.extend([
        wds.decode('pil'),
        to_tuple,
        map_tuple,
    ])
    dataset = wds.DataPipeline(
        *pipeline
    ).with_length(length)
    if end_epoch:
        dataset = dataset.with_epoch(end_epoch)
    if rc_batchsampler:
        dataset = dataset.compose(rc_batchsampler)
    return dataset


