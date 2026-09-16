# data/mock_data.py

MOCK_FRAMES = [
    # Frame 1
    [
        {
            "track_id": 1,
            "class_name": "car",
            "confidence": 0.95,
            "timestamp": 0.0,
            "bbox": [300, 200, 380, 320],
            "center_x": 340.0,
            "center_y": 260.0,
        },
        {
            "track_id": 2,
            "class_name": "bicycle",
            "confidence": 0.90,
            "timestamp": 0.0,
            "bbox": [100, 220, 150, 300],
            "center_x": 125.0,
            "center_y": 260.0,
        },
    ],

    # Frame 2
    [
        {
            "track_id": 1,
            "class_name": "car",
            "confidence": 0.96,
            "timestamp": 0.1,
            "bbox": [292, 192, 390, 332],
            "center_x": 341.0,
            "center_y": 262.0,
        },
        {
            "track_id": 2,
            "class_name": "bicycle",
            "confidence": 0.91,
            "timestamp": 0.1,
            "bbox": [105, 220, 155, 300],
            "center_x": 130.0,
            "center_y": 260.0,
        },
    ],

    # Frame 3
    [
        {
            "track_id": 1,
            "class_name": "car",
            "confidence": 0.96,
            "timestamp": 0.2,
            "bbox": [280, 180, 405, 350],
            "center_x": 342.5,
            "center_y": 265.0,
        },
        {
            "track_id": 2,
            "class_name": "bicycle",
            "confidence": 0.92,
            "timestamp": 0.2,
            "bbox": [110, 220, 160, 300],
            "center_x": 135.0,
            "center_y": 260.0,
        },
    ],
]