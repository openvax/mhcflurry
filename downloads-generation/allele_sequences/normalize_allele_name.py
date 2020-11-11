"""
Name normalization
"""

from mhcgnomes import parse, Allele, AlleleWithoutGene, Gene

def normalize_allele_name(raw_name):
    if any(item in raw_name.upper() for item in {"MIC", "HFE"}):
        print("Skipping %s, gene too different from Class Ia" % (
            raw_name,))
        return None
    result = parse(
        raw_name,
        preferred_result_types=(Allele,),
        infer_class2_pairing=True,
        required_result_types=(Allele, AlleleWithoutGene, Gene),
        raise_on_error=False)
    if result is None:
        print("Unable to parse as Class I allele or gene: %s" % raw_name)
        return None
    if not result.is_class1:
        print(
            "Skpping %s, wrong MHC class: %s" % (
                raw_name,
                result.mhc_class))
        return None
    if type(result) is Allele and (
            result.annotation_pseudogene or
            result.annotation_null or
            result.annotation_questionable):
        print("Skipping %s, due to annotation(s): %s" % (
            raw_name,
            result.annotations))
        return None
    name = result.to_string()
    print("Parsed '%s' as %s" % (raw_name, name))
    return name