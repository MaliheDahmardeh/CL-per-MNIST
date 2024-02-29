def get_permuted_mnist(args):
    assert not args.use_conv
    args.multiple_heads = False
    args.n_classes = 10
    if 'mem_size' in args:
        args.buffer_size = args.mem_size * args.n_classes
    args.n_tasks = 10 if args.n_tasks == -1 else args.n_tasks
    args.use_conv = False
    args.input_type = 'binary'
    args.input_size = [784]
    if args.output_loss is None:
        args.output_loss = 'bernoulli'

    # Fetch MNIST
    train = datasets.MNIST('Data/', train=True,  download=True)
    test  = datasets.MNIST('Data/', train=False, download=True)

    try:
        train_x, train_y = train.data, train.targets
        test_x, test_y = test.data, test.targets
    except:
        train_x, train_y = train.train_data, train.train_labels
        test_x, test_y = test.test_data, test.test_labels

    train_x = train_x.view(train_x.size(0), -1)
    test_x  = test_x.view(test_x.size(0), -1)

    train_ds, test_ds, inv_perms = [], [], []
    for task in range(args.n_tasks):
        perm = torch.arange(train_x.size(-1)) if task == 0 else torch.randperm(train_x.size(-1))

        # Build inverse permutations, so we can display samples
        inv_perm = torch.zeros_like(perm)
        for i in range(perm.size(0)):
            inv_perm[perm[i]] = i

        inv_perms += [inv_perm]
        train_ds  += [(train_x[:, perm], train_y)]
        test_ds   += [(test_x[:, perm],  test_y)]

    train_ds, val_ds = make_valid_from_train(train_ds)

    train_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'inv_perm': y, 'source': 'mnist'}), train_ds, inv_perms)
    val_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'inv_perm': y, 'source': 'mnist'}), val_ds, inv_perms)
    test_ds  = map(lambda x, y: XYDataset(x[0], x[1], **{'inv_perm': y, 'source': 'mnist'}), test_ds,  inv_perms)

    return train_ds, val_ds, test_ds

def make_valid_from_train(dataset, cut=0.95):
    tr_ds, val_ds = [], []
    for task_ds in dataset:
        x_t, y_t = task_ds

        # Shuffle before splitting
        perm = torch.randperm(len(x_t))
        x_t, y_t = x_t[perm], y_t[perm]

        split = int(len(x_t) * cut)
        x_tr, y_tr   = x_t[:split], y_t[:split]
        x_val, y_val = x_t[split:], y_t[split:]

        tr_ds  += [(x_tr, y_tr)]
        val_ds += [(x_val, y_val)]

    return tr_ds, val_ds
