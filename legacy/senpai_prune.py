def senpai_prune(parcellation, *args):
    """
    This function displays a GUI to assess and mark neural branches for pruning.
    Args:
        parcellation: numpy 3D array (uint8/uint16) containing a parcellation.
        nn (optional): Integer index of the first neuron to display (default=1).
        markers (optional): numpy 3D logical array for neural core masks.
    """
    N = np.max(parcellation)
    colormap = plt.cm.get_cmap('viridis', 101)
    nn = args[0] if len(args) > 0 else 1
    neuron = (parcellation == nn)
    markers = args[1] if len(args) > 1 else np.zeros_like(parcellation, dtype=bool)
    marker_regions = label(markers, connectivity=2)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    ax3d = fig.add_subplot(111, projection='3d')

    def plot_neuron():
        nonlocal ax3d
        ax3d.clear()
        verts, faces, _, _ = marching_cubes(neuron, level=0.5)
        colors = colormap((verts[:, 2] - verts[:, 2].min()) / (verts[:, 2].ptp() + 1e-6))
        mesh = Poly3DCollection(verts[faces], facecolors=colors, edgecolors='none', linewidth=0.2)
        ax3d.add_collection3d(mesh)
        ax3d.set_box_aspect([1, 1, 1])
        ax3d.set_xlim(0, neuron.shape[0])
        ax3d.set_ylim(0, neuron.shape[1])
        ax3d.set_zlim(0, neuron.shape[2])
        ax3d.view_init(30, 120)
        ax3d.axis("off")
        plt.draw()

    def update_view(new_nn):
        nonlocal neuron, nn
        nn = np.clip(new_nn, 1, N)
        neuron[:] = (parcellation == nn)
        neuron_label.set_text(f"Currently visualizing neuron {nn} of {N}")
        plot_neuron()

    def save_markers(event):
        with open("markers.pkl", "wb") as f:
            pickle.dump(markers, f)
        print("Markers saved to markers.pkl")

    def next_neuron(event):
        update_view(nn + 1)

    def prev_neuron(event):
        update_view(nn - 1)

    def goto_neuron(event):
        try:
            target_nn = int(neuron_index.text)
            update_view(target_nn)
        except ValueError:
            print("Invalid neuron index")

    def mark_branch(event):
        cursor = plt.ginput(1)
        if cursor:
            x, y, z = map(int, cursor[0])
            markers[x, y, z] = True
            print(f"Marked branch at: {x}, {y}, {z}")
            plot_neuron()

    # GUI elements
    neuron_label = plt.text(0.5, -0.1, f"Currently visualizing neuron {nn} of {N}", transform=ax.transAxes, ha='center')
    save_button = Button(plt.axes([0.75, 0.05, 0.15, 0.075]), 'Save')
    save_button.on_clicked(save_markers)
    next_button = Button(plt.axes([0.55, 0.05, 0.1, 0.075]), '>')
    next_button.on_clicked(next_neuron)
    prev_button = Button(plt.axes([0.35, 0.05, 0.1, 0.075]), '<')
    prev_button.on_clicked(prev_neuron)
    neuron_index = TextBox(plt.axes([0.4, 0.01, 0.1, 0.05]), 'Neuron #')
    go_button = Button(plt.axes([0.55, 0.01, 0.1, 0.05]), 'Go!')
    go_button.on_clicked(goto_neuron)
    mark_button = Button(plt.axes([0.75, 0.15, 0.15, 0.075]), 'Mark Branch')
    mark_button.on_clicked(mark_branch)

    update_view(nn)
    plt.show()