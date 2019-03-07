import pandas as pd
import numpy as np
from tqdm import tqdm

train_csv = pd.read_csv("train.csv")
test_csv = pd.read_csv("test.csv")

# fact_list = ['ProductName', 'EngineVersion', 'AppVersion',
#        'AvSigVersion', 'IsBeta', 'RtpStateBitfield', 'IsSxsPassiveMode',
#        'DefaultBrowsersIdentifier', 'AVProductStatesIdentifier',
#        'AVProductsInstalled', 'AVProductsEnabled', 'HasTpm',
#        'CountryIdentifier', 'CityIdentifier', 'OrganizationIdentifier',
#        'GeoNameIdentifier', 'LocaleEnglishNameIdentifier', 'Platform',
#        'Processor', 'OsVer', 'OsBuild', 'OsSuite', 'OsPlatformSubRelease',
#        'OsBuildLab', 'SkuEdition', 'IsProtected', 'AutoSampleOptIn', 'PuaMode',
#        'SMode', 'IeVerIdentifier', 'SmartScreen', 'Firewall', 'UacLuaenable',
#        'Census_MDC2FormFactor', 'Census_DeviceFamily',
#        'Census_OEMNameIdentifier', 'Census_OEMModelIdentifier',
#        'Census_ProcessorCoreCount', 'Census_ProcessorManufacturerIdentifier',
#        'Census_ProcessorModelIdentifier', 'Census_ProcessorClass',
#        'Census_PrimaryDiskTotalCapacity', 'Census_PrimaryDiskTypeName',
#        'Census_SystemVolumeTotalCapacity', 'Census_HasOpticalDiskDrive',
#        'Census_TotalPhysicalRAM', 'Census_ChassisTypeName',
#        'Census_InternalPrimaryDiagonalDisplaySizeInInches',
#        'Census_InternalPrimaryDisplayResolutionHorizontal',
#        'Census_InternalPrimaryDisplayResolutionVertical',
#        'Census_PowerPlatformRoleName', 'Census_InternalBatteryType',
#        'Census_InternalBatteryNumberOfCharges', 'Census_OSVersion',
#        'Census_OSArchitecture', 'Census_OSBranch', 'Census_OSBuildNumber',
#        'Census_OSBuildRevision', 'Census_OSEdition', 'Census_OSSkuName',
#        'Census_OSInstallTypeName', 'Census_OSInstallLanguageIdentifier',
#        'Census_OSUILocaleIdentifier', 'Census_OSWUAutoUpdateOptionsName',
#        'Census_IsPortableOperatingSystem', 'Census_GenuineStateName',
#        'Census_ActivationChannel', 'Census_IsFlightingInternal',
#        'Census_IsFlightsDisabled', 'Census_FlightRing',
#        'Census_ThresholdOptIn', 'Census_FirmwareManufacturerIdentifier',
#        'Census_FirmwareVersionIdentifier', 'Census_IsSecureBootEnabled',
#        'Census_IsWIMBootEnabled', 'Census_IsVirtualDevice',
#        'Census_IsTouchEnabled', 'Census_IsPenCapable',
#        'Census_IsAlwaysOnAlwaysConnectedCapable', 'Wdft_IsGamer',
#        'Wdft_RegionIdentifier']


def preprocess_to_numpy(train_csv,test_csv):
	train_csv = train_csv.replace(np.nan, '',regex=True)
	test_csv = test_csv.replace(np.nan, '',regex=True)

	train_labels = train_csv.HasDetections.to_numpy()
	test_machine_id = test_csv.MachineIdentifier.to_numpy()

	train_csv = train_csv.drop("HasDetections",axis=1)
	train_csv = train_csv.drop("MachineIdentifier",axis=1)
	test_csv = test_csv.drop("MachineIdentifier",axis=1)

	merged = pd.concat([train_csv,test_csv])
	category_info = []
	input_dimensions = []
	tt_split = len(train_csv)
	for c in tqdm(merged.columns):
		infoa, infob = pd.factorize(merged[c])
		merged[c] = infoa
		category_info.append(infob)
		input_dimensions.append(len(infob))
	merged = merged.to_numpy()
	train_csv = merged[:tt_split]
	test_csv = merged[tt_split:]
	np.save("train_data.npy",train_csv)
	np.save("test_data.npy",test_csv)
	np.save("train_labels.npy",train_labels)
	np.save("test_machine_id.npy",test_machine_id)

preprocess_to_numpy(train_csv,test_csv)
# train_np = preprocess_to_numpy(train_csv)
# np.save("train.npy",train_np)

# test_np = preprocess_to_numpy(test_csv)
# np.save("test.npy",test_np)

