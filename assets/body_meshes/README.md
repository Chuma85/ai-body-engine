# Body Mesh Assets

This folder is for local human base mesh assets used by the Blender synthetic rendering pipeline.

Large body assets are ignored by git. Do not commit licensed meshes, generated character files, or large binary assets to this repository.

Supported future formats:

- `.blend`
- `.fbx`
- `.obj`
- `.glb`
- `.gltf`

Suggested asset sources:

- MakeHuman exports
- MB-Lab generated mesh exports
- Custom Blender human meshes

Use `.env` values or render config paths to point to local assets. The Phase 2E example config references:

```text
assets/body_meshes/base_human.glb
```

If that file is not present and fallback is enabled, the renderer will use the procedural mannequin path instead.

## Phase 3H-B Blend Scene Guidance

The Phase 3H-B blend-file dataset workflow defaults to:

```text
assets/body_meshes/base_body_scene.blend
```

The scene must include `FrontCam`, `SideCam`, and `BackCam` cameras. The workflow can render front, side, and back PNGs directly from the `.blend` file and will write metadata describing the source blend file, camera set, seed, pose, and variation source.

If the scene has no shape keys, the workflow renders the static blend mesh and records `variation_source=static_blend_mesh`. True body shape variation requires shape keys, parametric mesh controls, or multiple body meshes.

## Phase 2G Rigged Mesh Guidance

For Phase 2G, the preferred local asset is:

```text
assets/body_meshes/base_human_rigged.fbx
```

Best sources for this phase:

- MakeHuman exported FBX files with a skeleton
- MB-Lab generated Blender characters
- Custom Blender human characters with armatures and shape keys

Static OBJ assets are still allowed, but they do not contain rigging or shape keys. The Phase 2G renderer will only use safe object-level scaling for static meshes so it avoids the tearing and seam artifacts caused by rough vertex-region deformation.

Do not commit large mesh assets. Keep them local and reference them from render configs.
