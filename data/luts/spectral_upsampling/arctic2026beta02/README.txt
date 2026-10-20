Arctic 2026 beta02 reflectance LUT

File:
  reflectance_xy_tc.npy

Source:
  arctic (Andrea Volpato), Arctic 2026 beta02 release, 2026-06-14
  https://discuss.pixls.us/t/spektrafilm-tech-discussions/57512/59
  Release archive: https://discuss.pixls.us/uploads/short-url/iYRpHaQWM3RVMYCoeaptnOu7j29.zip

The array is a 192 x 192 x 81 float16 reflectance LUT over tc coordinates
parameterizing the chromaticity triangle, sampled at 380-780 nm in 5 nm
steps. It was recovered under D65 and is relit by the selected film reference
illuminant before integration against film sensitivity. Reflectance values
are not bounded to [0, 1].

The LUT is distributed under CC BY-SA 4.0, consistent with the spektrafilm
project's published licensing for LUT assets. Preserve attribution to Andrea
Volpato and this source when redistributing or modifying it. This file is
included unchanged from the beta02 release archive.
