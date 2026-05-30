# Phase 5L Back-View Feature Summary

Dataset: `artifacts\phase_5l_back_view_shoulder_benchmark\dataset`
Samples: 360
Targets: shoulder_cm, across_back_cm, upper_back_cm, chest_cm, waist_cm, hip_cm, thigh_cm

Back-view features added:

- Back shoulder width proxy from shoulder peak width.
- Across-back proxy from upper-body maximum width.
- Upper-back width and area proxies.
- Shoulder slope proxy.
- Back torso width bands at shoulder, upper-chest, chest, mid-torso, waist, and hip levels.
- Front/back shoulder comparison and combined front/side/back volume proxies.

Back feature extractor: `back_view_silhouette_geometry_v1`
