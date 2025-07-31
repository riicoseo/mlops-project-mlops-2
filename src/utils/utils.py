def as_dict(obj):
    from tmdbv3api.as_obj import AsObj

    if isinstance(obj, AsObj):
        return {k: as_dict(v) for k, v in obj.__dict__.items()}

    if isinstance(obj, dict):
        # crew, cast 같은 경우 '_json' 내부에 리스트가 들어있으면 꺼내줌
        if '_json' in obj and isinstance(obj['_json'], list):
            return as_dict(obj['_json'])
        return {k: as_dict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [as_dict(v) for v in obj]

    return obj
