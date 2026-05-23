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
