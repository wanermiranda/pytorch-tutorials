from wompth.models.dqn import DQN, LayerConf, ScreenDims


def test_network_size():
    layout = [
        LayerConf(input=3, kernel_size=5, stride=2, batch_norm=16),
        LayerConf(input=16, kernel_size=5, stride=2, batch_norm=32),
        LayerConf(input=32, kernel_size=5, stride=2, batch_norm=32),
    ]
    dqn = DQN(layout=layout, screen_dims=ScreenDims(40, 90), outputs=2)

    assert dqn._linear_input_size == 512
    assert len(dqn._network_stack) == 6  # 3 convs, 3 norms
    assert dqn._head.in_features == 512 and dqn._head.out_features == 2
    assert (
        dqn.__str__()
        == "DQN(\n  (_Conv2d_0): Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2))\n  (_BatchNorm2d_0): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (_Conv2d_1): Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2))\n  (_BatchNorm2d_1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (_Conv2d_2): Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2))\n  (_BatchNorm2d_2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (_head): Linear(in_features=512, out_features=2, bias=True)\n)"
    )


def test_forward():
    layout = [
        LayerConf(input=3, kernel_size=5, stride=2, batch_norm=16),
        LayerConf(input=16, kernel_size=5, stride=2, batch_norm=32),
        LayerConf(input=32, kernel_size=5, stride=2, batch_norm=32),
    ]
    dqn = DQN(
        layout=layout,
        screen_dims=ScreenDims(40, 90),
        outputs=2,
    )
    dqn.eval()

    assert False
