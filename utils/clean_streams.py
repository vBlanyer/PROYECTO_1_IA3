def clean_streams(x):
      if isinstance(x, str):
            x = x.replace(",", "").strip()  # quitar comas y espacios
            if x.endswith("M"):  # millones
                  return float(x[:-1]) * 1e6
            elif x.endswith("B"):  # billones
                  return float(x[:-1]) * 1e9
            elif x.endswith("K"):  # miles
                  return float(x[:-1]) * 1e3
      try:
            return float(x)
      except:
            return 0
