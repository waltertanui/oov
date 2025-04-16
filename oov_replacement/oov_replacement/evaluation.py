def calculate_recognition_rate(system_oov_count: int, manual_oov_count: int) -> float:
    """
    Calculate the OOV word recognition rate.
    
    Args:
        system_oov_count (int): Number of OOV words detected by the system
        manual_oov_count (int): Number of OOV words manually counted
        
    Returns:
        float: Recognition rate as a percentage
    """
    if manual_oov_count == 0:
        return 0.0
    
    return (system_oov_count / manual_oov_count) * 100

def calculate_system_replacement_rate(correct_conversion_count: int, system_oov_count: int) -> float:
    """
    Calculate the system OOV word replacement rate.
    
    Args:
        correct_conversion_count (int): Number of correctly replaced OOV words
        system_oov_count (int): Number of OOV words detected by the system
        
    Returns:
        float: System replacement rate as a percentage
    """
    if system_oov_count == 0:
        return 0.0
    
    return (correct_conversion_count / system_oov_count) * 100

def calculate_replacement_rate(correct_conversion_count: int, manual_oov_count: int) -> float:
    """
    Calculate the overall OOV word replacement rate.
    
    Args:
        correct_conversion_count (int): Number of correctly replaced OOV words
        manual_oov_count (int): Number of OOV words manually counted
        
    Returns:
        float: Replacement rate as a percentage
    """
    if manual_oov_count == 0:
        return 0.0
    
    return (correct_conversion_count / manual_oov_count) * 100