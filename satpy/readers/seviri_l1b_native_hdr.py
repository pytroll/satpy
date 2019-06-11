#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2019 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Header and trailer records of SEVIRI native format.
"""

import numpy as np

from satpy.readers.eum_base import (time_cds_short, time_cds,
                                    time_cds_expanded)


class GSDTRecords(object):
    """MSG Ground Segment Data Type records.

    Reference Document (EUM/MSG/SPE/055):
    MSG Ground Segment Design Specification (GSDS)

    """
    gp_fac_env = np.uint8
    gp_fac_id = np.uint8
    gp_sc_id = np.uint16
    gp_su_id = np.uint32
    gp_svce_type = np.uint8

    # 4 bytes
    gp_cpu_address = [
        ('Qualifier_1', np.uint8),
        ('Qualifier_2', np.uint8),
        ('Qualifier_3', np.uint8),
        ('Qualifier_4', np.uint8)
    ]

    # 22 bytes
    gp_pk_header = [
        ('HeaderVersionNo', np.uint8),
        ('PacketType', np.uint8),
        ('SubHeaderType', np.uint8),
        ('SourceFacilityId', gp_fac_id),
        ('SourceEnvId', gp_fac_env),
        ('SourceInstanceId', np.uint8),
        ('SourceSUId', gp_su_id),
        ('SourceCPUId', gp_cpu_address),
        ('DestFacilityId', gp_fac_id),
        ('DestEnvId', gp_fac_env),
        ('SequenceCount', np.uint16),
        ('PacketLength', np.int32)
    ]

    # 16 bytes
    gp_pk_sh1 = [
        ('SubHeaderVersionNo', np.uint8),
        ('ChecksumFlag', np.bool),
        ('Acknowledgement', (np.uint8, 4)),
        ('ServiceType', gp_svce_type),
        ('ServiceSubtype', np.uint8),
        ('PacketTime', time_cds_short),
        ('SpacecraftId', gp_sc_id)
    ]


class Msg15NativeHeaderRecord(object):
    """
    SEVIRI Level 1.5 header for native-format
    """

    def get(self):

        # 450400 bytes
        record = [
            ('15_MAIN_PRODUCT_HEADER', L15MainProductHeaderRecord().get()),
            ('15_SECONDARY_PRODUCT_HEADER',
             L15SecondaryProductHeaderRecord().get()),
            ('GP_PK_HEADER', GSDTRecords.gp_pk_header),
            ('GP_PK_SH1', GSDTRecords.gp_pk_sh1),
            ('15_DATA_HEADER', L15DataHeaderRecord().get())
        ]

        return np.dtype(record).newbyteorder('>')


class L15PhData(object):

    # 80 bytes
    l15_ph_data = [
        ('Name', 'S30'),
        ('Value', 'S50')
    ]


class L15MainProductHeaderRecord(object):
    """
    Reference Document:
    MSG Level 1.5 Native Format File Definition
    """

    def get(self):

        l15_ph_data = L15PhData.l15_ph_data

        l15_ph_data_identification = [
            ('Name', 'S30'),
            ('Size', 'S16'),
            ('Address', 'S16')
        ]

        # 3674 bytes
        record = [
            ('FormatName', l15_ph_data),
            ('FormatDocumentName', l15_ph_data),
            ('FormatDocumentMajorVersion', l15_ph_data),
            ('FormatDocumentMinorVersion', l15_ph_data),
            ('CreationDateTime', l15_ph_data),
            ('CreatingCentre', l15_ph_data),
            ('DataSetIdentification', (l15_ph_data_identification, 27)),
            ('TotalFileSize', l15_ph_data),
            ('GORT', l15_ph_data),
            ('ASTI', l15_ph_data),
            ('LLOS', l15_ph_data),
            ('SNIT', l15_ph_data),
            ('AIID', l15_ph_data),
            ('SSBT', l15_ph_data),
            ('SSST', l15_ph_data),
            ('RRCC', l15_ph_data),
            ('RRBT', l15_ph_data),
            ('RRST', l15_ph_data),
            ('PPRC', l15_ph_data),
            ('PPDT', l15_ph_data),
            ('GPLV', l15_ph_data),
            ('APNM', l15_ph_data),
            ('AARF', l15_ph_data),
            ('UUDT', l15_ph_data),
            ('QQOV', l15_ph_data),
            ('UDSP', l15_ph_data)
        ]

        return record


class L15SecondaryProductHeaderRecord(object):
    """
    Reference Document:
    MSG Level 1.5 Native Format File Definition
    """

    def get(self):

        l15_ph_data = L15PhData.l15_ph_data

        # 1440 bytes
        record = [
            ('ABID', l15_ph_data),
            ('SMOD', l15_ph_data),
            ('APXS', l15_ph_data),
            ('AVPA', l15_ph_data),
            ('LSCD', l15_ph_data),
            ('LMAP', l15_ph_data),
            ('QDLC', l15_ph_data),
            ('QDLP', l15_ph_data),
            ('QQAI', l15_ph_data),
            ('SelectedBandIDs', l15_ph_data),
            ('SouthLineSelectedRectangle', l15_ph_data),
            ('NorthLineSelectedRectangle', l15_ph_data),
            ('EastColumnSelectedRectangle', l15_ph_data),
            ('WestColumnSelectedRectangle', l15_ph_data),
            ('NumberLinesVISIR', l15_ph_data),
            ('NumberColumnsVISIR', l15_ph_data),
            ('NumberLinesHRV', l15_ph_data),
            ('NumberColumnsHRV', l15_ph_data)
        ]

        return record


class L15DataHeaderRecord(object):
    """
    Reference Document (EUM/MSG/ICD/105):
    MSG Level 1.5 Image Data Format Description
    """

    def get(self):

        # 445248 bytes
        record = [
            ('15HeaderVersion', np.uint8),
            ('SatelliteStatus', self.satellite_status),
            ('ImageAcquisition', self.image_acquisition),
            ('CelestialEvents', self.celestial_events),
            ('ImageDescription', self.image_description),
            ('RadiometricProcessing', self.radiometric_processing),
            ('GeometricProcessing', self.geometric_processing),
            ('IMPFConfiguration', self.impf_configuration)]

        return record

    @property
    def satellite_status(self):

        # 7 bytes
        satellite_definition = [
            ('SatelliteId', np.uint16),
            ('NominalLongitude', np.float32),
            ('SatelliteStatus', np.uint8)]

        # 28 bytes
        satellite_operations = [
            ('LastManoeuvreFlag', np.bool),
            ('LastManoeuvreStartTime', time_cds_short),
            ('LastManoeuvreEndTime', time_cds_short),
            ('LastManoeuvreType', np.uint8),
            ('NextManoeuvreFlag', np.bool),
            ('NextManoeuvreStartTime', time_cds_short),
            ('NextManoeuvreEndTime', time_cds_short),
            ('NextManoeuvreType', np.uint8)]

        # 396 bytes
        orbit_coeff = [
            ('StartTime', time_cds_short),
            ('EndTime', time_cds_short),
            ('X', (np.float64, 8)),
            ('Y', (np.float64, 8)),
            ('Z', (np.float64, 8)),
            ('VX', (np.float64, 8)),
            ('VY', (np.float64, 8)),
            ('VZ', (np.float64, 8))]

        # 39612 bytes
        orbit = [
            ('PeriodStartTime', time_cds_short),
            ('PeriodEndTime', time_cds_short),
            ('OrbitPolynomial', (orbit_coeff, 100))]

        # 204 bytes
        attitude_coeff = [
            ('StartTime', time_cds_short),
            ('EndTime', time_cds_short),
            ('XofSpinAxis', (np.float64, 8)),
            ('YofSpinAxis', (np.float64, 8)),
            ('ZofSpinAxis', (np.float64, 8))]

        # 20420 bytes
        attitude = [
            ('PeriodStartTime', time_cds_short),
            ('PeriodEndTime', time_cds_short),
            ('PrincipleAxisOffsetAngle', np.float64),
            ('AttitudePolynomial', (attitude_coeff, 100))]

        # 59 bytes
        utc_correlation = [
            ('PeriodStartTime', time_cds_short),
            ('PeriodEndTime', time_cds_short),
            ('OnBoardTimeStart', (np.uint8, 7)),
            ('VarOnBoardTimeStart', np.float64),
            ('A1', np.float64),
            ('VarA1', np.float64),
            ('A2', np.float64),
            ('VarA2', np.float64)]

        # 60134 bytes
        record = [
            ('SatelliteDefinition', satellite_definition),
            ('SatelliteOperations', satellite_operations),
            ('Orbit', orbit),
            ('Attitude', attitude),
            ('SpinRetreatRCStart', np.float64),
            ('UTCCorrelation', utc_correlation)]

        return record

    @property
    def image_acquisition(self):

        planned_acquisition_time = [
            ('TrueRepeatCycleStart', time_cds_expanded),
            ('PlanForwardScanEnd', time_cds_expanded),
            ('PlannedRepeatCycleEnd', time_cds_expanded)]

        radiometer_status = [
            ('ChannelStatus', (np.uint8, 12)),
            ('DetectorStatus', (np.uint8, 42))]

        hrv_frame_offsets = [
            ('MDUNomHRVDelay1', np.uint16),
            ('MDUNomHRVDelay2', np.uint16),
            ('Spare', np.uint16),
            ('MDUNomHRVBreakLine', np.uint16)]

        operation_parameters = [
            ('L0_LineCounter', np.uint16),
            ('K1_RetraceLines', np.uint16),
            ('K2_PauseDeciseconds', np.uint16),
            ('K3_RetraceLines', np.uint16),
            ('K4_PauseDeciseconds', np.uint16),
            ('K5_RetraceLines', np.uint16),
            ('XDeepSpaceWindowPosition', np.uint8)]

        radiometer_settings = [
            ('MDUSamplingDelays', (np.uint16, 42)),
            ('HRVFrameOffsets', hrv_frame_offsets),
            ('DHSSSynchSelection', np.uint8),
            ('MDUOutGain', (np.uint16, 42)),
            ('MDUCoarseGain', (np.uint8, 42)),
            ('MDUFineGain', (np.uint16, 42)),
            ('MDUNumericalOffset', (np.uint16, 42)),
            ('PUGain', (np.uint16, 42)),
            ('PUOffset', (np.uint16, 27)),
            ('PUBias', (np.uint16, 15)),
            ('OperationParameters', operation_parameters),
            ('RefocusingLines', np.uint16),
            ('RefocusingDirection', np.uint8),
            ('RefocusingPosition', np.uint16),
            ('ScanRefPosFlag', np.bool),
            ('ScanRefPosNumber', np.uint16),
            ('ScanRefPosVal', np.float32),
            ('ScanFirstLine', np.uint16),
            ('ScanLastLine', np.uint16),
            ('RetraceStartLine', np.uint16)]

        decontamination = [
            ('DecontaminationNow', np.bool),
            ('DecontaminationStart', time_cds_short),
            ('DecontaminationEnd', time_cds_short)]

        radiometer_operations = [
            ('LastGainChangeFlag', np.bool),
            ('LastGainChangeTime', time_cds_short),
            ('Decontamination', decontamination),
            ('BBCalScheduled', np.bool),
            ('BBCalibrationType', np.uint8),
            ('BBFirstLine', np.uint16),
            ('BBLastLine', np.uint16),
            ('ColdFocalPlaneOpTemp', np.uint16),
            ('WarmFocalPlaneOpTemp', np.uint16)]

        record = [
            ('PlannedAcquisitionTime', planned_acquisition_time),
            ('RadiometerStatus', radiometer_status),
            ('RadiometerSettings', radiometer_settings),
            ('RadiometerOperations', radiometer_operations)]

        return record

    @property
    def celestial_events(self):

        earth_moon_sun_coeff = [
            ('StartTime', time_cds_short),
            ('EndTime', time_cds_short),
            ('AlphaCoef', (np.float64, 8)),
            ('BetaCoef', (np.float64, 8))]

        star_coeff = [
            ('StarId', np.uint16),
            ('StartTime', time_cds_short),
            ('EndTime', time_cds_short),
            ('AlphaCoef', (np.float64, 8)),
            ('BetaCoef', (np.float64, 8))]

        ephemeris = [
            ('PeriodTimeStart', time_cds_short),
            ('PeriodTimeEnd', time_cds_short),
            ('RelatedOrbitFileTime', 'S15'),
            ('RelatedAttitudeFileTime', 'S15'),
            ('EarthEphemeris', (earth_moon_sun_coeff, 100)),
            ('MoonEphemeris', (earth_moon_sun_coeff, 100)),
            ('SunEphemeris', (earth_moon_sun_coeff, 100)),
            ('StarEphemeris', (star_coeff, (20, 100)))]

        relation_to_image = [
            ('TypeOfEclipse', np.uint8),
            ('EclipseStartTime', time_cds_short),
            ('EclipseEndTime', time_cds_short),
            ('VisibleBodiesInImage', np.uint8),
            ('BodiesCloseToFOV', np.uint8),
            ('ImpactOnImageQuality', np.uint8)]

        record = [
            ('CelestialBodiesPosition', ephemeris),
            ('RelationToImage', relation_to_image)]

        return record

    @property
    def image_description(self):

        projection_description = [
            ('TypeOfProjection', np.uint8),
            ('LongitudeOfSSP', np.float32)]

        reference_grid = [
            ('NumberOfLines', np.int32),
            ('NumberOfColumns', np.int32),
            ('LineDirGridStep', np.float32),
            ('ColumnDirGridStep', np.float32),
            ('GridOrigin', np.uint8)]

        planned_coverage_vis_ir = [
            ('SouthernLinePlanned', np.int32),
            ('NorthernLinePlanned', np.int32),
            ('EasternColumnPlanned', np.int32),
            ('WesternColumnPlanned', np.int32)]

        planned_coverage_hrv = [
            ('LowerSouthLinePlanned', np.int32),
            ('LowerNorthLinePlanned', np.int32),
            ('LowerEastColumnPlanned', np.int32),
            ('LowerWestColumnPlanned', np.int32),
            ('UpperSouthLinePlanned', np.int32),
            ('UpperNorthLinePlanned', np.int32),
            ('UpperEastColumnPlanned', np.int32),
            ('UpperWestColumnPlanned', np.int32)]

        level_15_image_production = [
            ('ImageProcDirection', np.uint8),
            ('PixelGenDirection', np.uint8),
            ('PlannedChanProcessing', (np.uint8, 12))]

        record = [
            ('ProjectionDescription', projection_description),
            ('ReferenceGridVIS_IR', reference_grid),
            ('ReferenceGridHRV', reference_grid),
            ('PlannedCoverageVIS_IR', planned_coverage_vis_ir),
            ('PlannedCoverageHRV', planned_coverage_hrv),
            ('Level15ImageProduction', level_15_image_production)]

        return record

    @property
    def radiometric_processing(self):

        rp_summary = [
            ('RadianceLinearization', (np.bool, 12)),
            ('DetectorEqualization', (np.bool, 12)),
            ('OnboardCalibrationResult', (np.bool, 12)),
            ('MPEFCalFeedback', (np.bool, 12)),
            ('MTFAdaptation', (np.bool, 12)),
            ('StrayLightCorrection', (np.bool, 12))]

        level_15_image_calibration = [
            ('CalSlope', np.float64),
            ('CalOffset', np.float64)]

        time_cuc_size = [
            ('CT1', np.uint8),
            ('CT2', np.uint8),
            ('CT3', np.uint8),
            ('CT4', np.uint8),
            ('FT1', np.uint8),
            ('FT2', np.uint8),
            ('FT3', np.uint8)]

        cold_fp_temperature = [
            ('FCUNominalColdFocalPlaneTemp', np.uint16),
            ('FCURedundantColdFocalPlaneTemp', np.uint16)]

        warm_fp_temperature = [
            ('FCUNominalWarmFocalPlaneVHROTemp', np.uint16),
            ('FCURedundantWarmFocalPlaneVHROTemp', np.uint16)]

        scan_mirror_temperature = [
            ('FCUNominalScanMirrorSensor1Temp', np.uint16),
            ('FCURedundantScanMirrorSensor1Temp', np.uint16),
            ('FCUNominalScanMirrorSensor2Temp', np.uint16),
            ('FCURedundantScanMirrorSensor2Temp', np.uint16)]

        m1m2m3_temperature = [
            ('FCUNominalM1MirrorSensor1Temp', np.uint16),
            ('FCURedundantM1MirrorSensor1Temp', np.uint16),
            ('FCUNominalM1MirrorSensor2Temp', np.uint16),
            ('FCURedundantM1MirrorSensor2Temp', np.uint16),
            ('FCUNominalM23AssemblySensor1Temp', np.uint8),
            ('FCURedundantM23AssemblySensor1Temp', np.uint8),
            ('FCUNominalM23AssemblySensor2Temp', np.uint8),
            ('FCURedundantM23AssemblySensor2Temp', np.uint8)]

        baffle_temperature = [
            ('FCUNominalM1BaffleTemp', np.uint16),
            ('FCURedundantM1BaffleTemp', np.uint16)]

        blackbody_temperature = [
            ('FCUNominalBlackBodySensorTemp', np.uint16),
            ('FCURedundantBlackBodySensorTemp', np.uint16)]

        fcu_mode = [
            ('FCUNominalSMMStatus', 'S2'),
            ('FCURedundantSMMStatus', 'S2')]

        extracted_bb_data = [
            ('NumberOfPixelsUsed', np.uint32),
            ('MeanCount', np.float32),
            ('RMS', np.float32),
            ('MaxCount', np.uint16),
            ('MinCount', np.uint16),
            ('BB_Processing_Slope', np.float64),
            ('BB_Processing_Offset', np.float64)]

        bb_related_data = [
            ('OnBoardBBTime', time_cuc_size),
            ('MDUOutGain', (np.uint16, 42)),
            ('MDUCoarseGain', (np.uint8, 42)),
            ('MDUFineGain', (np.uint16, 42)),
            ('MDUNumericalOffset', (np.uint16, 42)),
            ('PUGain', (np.uint16, 42)),
            ('PUOffset', (np.uint16, 27)),
            ('PUBias', (np.uint16, 15)),
            ('DCRValues', (np.uint8, 63)),
            ('X_DeepSpaceWindowPosition', np.int8),
            ('ColdFPTemperature', cold_fp_temperature),
            ('WarmFPTemperature', warm_fp_temperature),
            ('ScanMirrorTemperature', scan_mirror_temperature),
            ('M1M2M3Temperature', m1m2m3_temperature),
            ('BaffleTemperature', baffle_temperature),
            ('BlackBodyTemperature', blackbody_temperature),
            ('FCUMode', fcu_mode),
            ('ExtractedBBData', (extracted_bb_data, 12))]

        black_body_data_used = [
            ('BBObservationUTC', time_cds_expanded),
            ('BBRelatedData', bb_related_data)]

        impf_cal_data = [
            ('ImageQualityFlag', np.uint8),
            ('ReferenceDataFlag', np.uint8),
            ('AbsCalMethod', np.uint8),
            ('Pad1', 'S1'),
            ('AbsCalWeightVic', np.float32),
            ('AbsCalWeightXsat', np.float32),
            ('AbsCalCoeff', np.float32),
            ('AbsCalError', np.float32),
            ('GSICSCalCoeff', np.float32),
            ('GSICSCalError', np.float32),
            ('GSICSOffsetCount', np.float32)]

        rad_proc_mtf_adaptation = [
            ('VIS_IRMTFCorrectionE_W', (np.float32, (33, 16))),
            ('VIS_IRMTFCorrectionN_S', (np.float32, (33, 16))),
            ('HRVMTFCorrectionE_W', (np.float32, (9, 16))),
            ('HRVMTFCorrectionN_S', (np.float32, (9, 16))),
            ('StraylightCorrection', (np.float32, (12, 8, 8)))]

        record = [
            ('RPSummary', rp_summary),
            ('Level15ImageCalibration', (level_15_image_calibration, 12)),
            ('BlackBodyDataUsed', black_body_data_used),
            ('MPEFCalFeedback', (impf_cal_data, 12)),
            ('RadTransform', (np.float32, (42, 64))),
            ('RadProcMTFAdaptation', rad_proc_mtf_adaptation)]

        return record

    @property
    def geometric_processing(self):

        opt_axis_distances = [
            ('E-WFocalPlane', (np.float32, 42)),
            ('N_SFocalPlane', (np.float32, 42))]

        earth_model = [
            ('TypeOfEarthModel', np.uint8),
            ('EquatorialRadius', np.float64),
            ('NorthPolarRadius', np.float64),
            ('SouthPolarRadius', np.float64)]

        record = [
            ('OptAxisDistances', opt_axis_distances),
            ('EarthModel', earth_model),
            ('AtmosphericModel', (np.float32, (12, 360))),
            ('ResamplingFunctions', (np.uint8, 12))]

        return record

    @property
    def impf_configuration(self):

        overall_configuration = [
            ('Issue', np.uint16),
            ('Revision', np.uint16)
        ]

        sw_version = overall_configuration

        info_base_versions = sw_version

        su_configuration = [
            ('SWVersion', sw_version),
            ('InfoBaseVersions', (info_base_versions, 10))
        ]

        su_details = [
            ('SUId', GSDTRecords.gp_su_id),
            ('SUIdInstance', np.int8),
            ('SUMode', np.uint8),
            ('SUState', np.uint8),
            ('SUConfiguration', su_configuration)
        ]

        equalisation_params = [
            ('ConstCoeff', np.float32),
            ('LinearCoeff', np.float32),
            ('QuadraticCoeff', np.float32)
        ]

        black_body_data_for_warm_start = [
            ('GTotalForMethod1', (np.float64, 12)),
            ('GTotalForMethod2', (np.float64, 12)),
            ('GTotalForMethod3', (np.float64, 12)),
            ('GBackForMethod1', (np.float64, 12)),
            ('GBackForMethod2', (np.float64, 12)),
            ('GBackForMethod3', (np.float64, 12)),
            ('RatioGTotalToGBack', (np.float64, 12)),
            ('GainInFrontOpticsCont', (np.float64, 12)),
            ('CalibrationConstants', (np.float32, 12)),
            ('maxIncidentRadiance', (np.float64, 12)),
            ('TimeOfColdObsSeconds', np.float64),
            ('TimeOfColdObsNanoSecs', np.float64),
            ('IncidenceRadiance', (np.float64, 12)),
            ('TempCal', np.float64),
            ('TempM1', np.float64),
            ('TempScan', np.float64),
            ('TempM1Baf', np.float64),
            ('TempCalSurround', np.float64)
        ]

        mirror_parameters = [
            ('MaxFeedbackVoltage', np.float64),
            ('MinFeedbackVoltage', np.float64),
            ('MirrorSlipEstimate', np.float64)
        ]

        hktm_parameters = [
            ('TimeS0Packet', time_cds_short),
            ('TimeS1Packet', time_cds_short),
            ('TimeS2Packet', time_cds_short),
            ('TimeS3Packet', time_cds_short),
            ('TimeS4Packet', time_cds_short),
            ('TimeS5Packet', time_cds_short),
            ('TimeS6Packet', time_cds_short),
            ('TimeS7Packet', time_cds_short),
            ('TimeS8Packet', time_cds_short),
            ('TimeS9Packet', time_cds_short),
            ('TimeSYPacket', time_cds_short),
            ('TimePSPacket', time_cds_short)
        ]

        warm_start_params = [
            ('ScanningLaw', (np.float64, 1527)),
            ('RadFramesAlignment', (np.float64, 3)),
            ('ScanningLawVariation', (np.float32, 2)),
            ('EqualisationParams', (equalisation_params, 42)),
            ('BlackBodyDataForWarmStart', black_body_data_for_warm_start),
            ('MirrorParameters', mirror_parameters),
            ('LastSpinPeriod', np.float64),
            ('HKTMParameters', hktm_parameters),
            ('WSPReserved', (np.uint8, 3312))
        ]

        record = [
            ('OverallConfiguration', overall_configuration),
            ('SUDetails', (su_details, 50)),
            ('WarmStartParams', warm_start_params)
        ]

        return record


class Msg15NativeTrailerRecord(object):
    """
    SEVIRI Level 1.5 trailer for native-format

    Reference Document (EUM/MSG/ICD/105):
    MSG Level 1.5 Image Data Format Description
    """

    def get(self):

        # 380363 bytes
        record = [
            ('GP_PK_HEADER', GSDTRecords.gp_pk_header),
            ('GP_PK_SH1', GSDTRecords.gp_pk_sh1),
            ('15TRAILER', self.seviri_l15_trailer)
        ]

        return np.dtype(record).newbyteorder('>')

    @property
    def seviri_l15_trailer(self):

        record = [
            ('15TrailerVersion', np.uint8),
            ('ImageProductionStats', self.image_production_stats),
            ('NavigationExtractionResults', self.navigation_extraction_results),
            ('RadiometricQuality', self.radiometric_quality),
            ('GeometricQuality', self.geometric_quality),
            ('TimelinessAndCompleteness', self.timeliness_and_completeness)
        ]

        return record

    @property
    def image_production_stats(self):

        gp_sc_id = GSDTRecords.gp_sc_id

        actual_scanning_summary = [
            ('NominalImageScanning', np.uint8),
            ('ReducedScan', np.uint8),
            ('ForwardScanStart', time_cds_short),
            ('ForwardScanEnd', time_cds_short)
        ]

        radiometric_behaviour = [
            ('NominalBehaviour', np.uint8),
            ('RadScanIrregularity', np.uint8),
            ('RadStoppage', np.uint8),
            ('RepeatCycleNotCompleted', np.uint8),
            ('GainChangeTookPlace', np.uint8),
            ('DecontaminationTookPlace', np.uint8),
            ('NoBBCalibrationAchieved', np.uint8),
            ('IncorrectTemperature', np.uint8),
            ('InvalidBBData', np.uint8),
            ('InvalidAuxOrHKTMData', np.uint8),
            ('RefocusingMechanismActuated', np.uint8),
            ('MirrorBackToReferencePos', np.uint8)
        ]

        reception_summary_stats = [
            ('PlannedNumberOfL10Lines', (np.uint32, 12)),
            ('NumberOfMissingL10Lines', (np.uint32, 12)),
            ('NumberOfCorruptedL10Lines', (np.uint32, 12)),
            ('NumberOfReplacedL10Lines', (np.uint32, 12))
        ]

        l15_image_validity = [
            ('NominalImage', np.uint8),
            ('NonNominalBecauseIncomplete', np.uint8),
            ('NonNominalRadiometricQuality', np.uint8),
            ('NonNominalGeometricQuality', np.uint8),
            ('NonNominalTimeliness', np.uint8),
            ('IncompleteL15', np.uint8),
        ]

        actual_l15_coverage_vis_ir = [
            ('SouthernLineActual', np.int32),
            ('NorthernLineActual', np.int32),
            ('EasternColumnActual', np.int32),
            ('WesternColumnActual', np.int32)
        ]

        actual_l15_coverage_hrv = [
            ('LowerSouthLineActual', np.int32),
            ('LowerNorthLineActual', np.int32),
            ('LowerEastColumnActual', np.int32),
            ('LowerWestColumnActual', np.int32),
            ('UpperSouthLineActual', np.int32),
            ('UpperNorthLineActual', np.int32),
            ('UpperEastColumnActual', np.int32),
            ('UpperWestColumnActual', np.int32),
        ]

        record = [
            ('SatelliteId', gp_sc_id),
            ('ActualScanningSummary', actual_scanning_summary),
            ('RadiometricBehaviour', radiometric_behaviour),
            ('ReceptionSummaryStats', reception_summary_stats),
            ('L15ImageValidity', (l15_image_validity, 12)),
            ('ActualL15CoverageVIS_IR', actual_l15_coverage_vis_ir),
            ('ActualL15CoverageHRV', actual_l15_coverage_hrv)
        ]

        return record

    @property
    def navigation_extraction_results(self):

        horizon_observation = [
            ('HorizonId', np.uint8),
            ('Alpha', np.float64),
            ('AlphaConfidence', np.float64),
            ('Beta', np.float64),
            ('BetaConfidence', np.float64),
            ('ObservationTime', time_cds),
            ('SpinRate', np.float64),
            ('AlphaDeviation', np.float64),
            ('BetaDeviation', np.float64)
        ]

        star_observation = [
            ('StarId', np.uint16),
            ('Alpha', np.float64),
            ('AlphaConfidence', np.float64),
            ('Beta', np.float64),
            ('BetaConfidence', np.float64),
            ('ObservationTime', time_cds),
            ('SpinRate', np.float64),
            ('AlphaDeviation', np.float64),
            ('BetaDeviation', np.float64)
        ]

        landmark_observation = [
            ('LandmarkId', np.uint16),
            ('LandmarkLongitude', np.float64),
            ('LandmarkLatitude', np.float64),
            ('Alpha', np.float64),
            ('AlphaConfidence', np.float64),
            ('Beta', np.float64),
            ('BetaConfidence', np.float64),
            ('ObservationTime', time_cds),
            ('SpinRate', np.float64),
            ('AlphaDeviation', np.float64),
            ('BetaDeviation', np.float64)
        ]

        record = [
            ('ExtractedHorizons', (horizon_observation, 4)),
            ('ExtractedStars', (star_observation, 20)),
            ('ExtractedLandmarks', (landmark_observation, 50))
        ]

        return record

    @property
    def radiometric_quality(self):

        l10_rad_quality = [
            ('FullImageMinimumCount', np.uint16),
            ('FullImageMaximumCount', np.uint16),
            ('EarthDiskMinimumCount', np.uint16),
            ('EarthDiskMaximumCount', np.uint16),
            ('MoonMinimumCount', np.uint16),
            ('MoonMaximumCount', np.uint16),
            ('FullImageMeanCount', np.float32),
            ('FullImageStandardDeviation', np.float32),
            ('EarthDiskMeanCount', np.float32),
            ('EarthDiskStandardDeviation', np.float32),
            ('MoonMeanCount', np.float32),
            ('MoonStandardDeviation', np.float32),
            ('SpaceMeanCount', np.float32),
            ('SpaceStandardDeviation', np.float32),
            ('SESpaceCornerMeanCount', np.float32),
            ('SESpaceCornerStandardDeviation', np.float32),
            ('SWSpaceCornerMeanCount', np.float32),
            ('SWSpaceCornerStandardDeviation', np.float32),
            ('NESpaceCornerMeanCount', np.float32),
            ('NESpaceCornerStandardDeviation', np.float32),
            ('NWSpaceCornerMeanCount', np.float32),
            ('NWSpaceCornerStandardDeviation', np.float32),
            ('4SpaceCornersMeanCount', np.float32),
            ('4SpaceCornersStandardDeviation', np.float32),
            ('FullImageHistogram', (np.uint32, 256)),
            ('EarthDiskHistogram', (np.uint32, 256)),
            ('ImageCentreSquareHistogram', (np.uint32, 256)),
            ('SESpaceCornerHistogram', (np.uint32, 128)),
            ('SWSpaceCornerHistogram', (np.uint32, 128)),
            ('NESpaceCornerHistogram', (np.uint32, 128)),
            ('NWSpaceCornerHistogram', (np.uint32, 128)),
            ('FullImageEntropy', (np.float32, 3)),
            ('EarthDiskEntropy', (np.float32, 3)),
            ('ImageCentreSquareEntropy', (np.float32, 3)),
            ('SESpaceCornerEntropy', (np.float32, 3)),
            ('SWSpaceCornerEntropy', (np.float32, 3)),
            ('NESpaceCornerEntropy', (np.float32, 3)),
            ('NWSpaceCornerEntropy', (np.float32, 3)),
            ('4SpaceCornersEntropy', (np.float32, 3)),
            ('ImageCentreSquarePSD_EW', (np.float32, 128)),
            ('FullImagePSD_EW', (np.float32, 128)),
            ('ImageCentreSquarePSD_NS', (np.float32, 128)),
            ('FullImagePSD_NS', (np.float32, 128))
        ]

        l15_rad_quality = [
            ('FullImageMinimumCount', np.uint16),
            ('FullImageMaximumCount', np.uint16),
            ('EarthDiskMinimumCount', np.uint16),
            ('EarthDiskMaximumCount', np.uint16),
            ('FullImageMeanCount', np.float32),
            ('FullImageStandardDeviation', np.float32),
            ('EarthDiskMeanCount', np.float32),
            ('EarthDiskStandardDeviation', np.float32),
            ('SpaceMeanCount', np.float32),
            ('SpaceStandardDeviation', np.float32),
            ('FullImageHistogram', (np.uint32, 256)),
            ('EarthDiskHistogram', (np.uint32, 256)),
            ('ImageCentreSquareHistogram', (np.uint32, 256)),
            ('FullImageEntropy', (np.float32, 3)),
            ('EarthDiskEntropy', (np.float32, 3)),
            ('ImageCentreSquareEntropy', (np.float32, 3)),
            ('ImageCentreSquarePSD_EW', (np.float32, 128)),
            ('FullImagePSD_EW', (np.float32, 128)),
            ('ImageCentreSquarePSD_NS', (np.float32, 128)),
            ('FullImagePSD_NS', (np.float32, 128)),
            ('SESpaceCornerL15_RMS', np.float32),
            ('SESpaceCornerL15_Mean', np.float32),
            ('SWSpaceCornerL15_RMS', np.float32),
            ('SWSpaceCornerL15_Mean', np.float32),
            ('NESpaceCornerL15_RMS', np.float32),
            ('NESpaceCornerL15_Mean', np.float32),
            ('NWSpaceCornerL15_RMS', np.float32),
            ('NWSpaceCornerL15_Mean', np.float32)
        ]

        record = [
            ('L10RadQuality', (l10_rad_quality, 42)),
            ('L15RadQuality', (l15_rad_quality, 12))
        ]

        return record

    @property
    def geometric_quality(self):

        absolute_accuracy = [
            ('QualityInfoValidity', np.uint8),
            ('EastWestAccuracyRMS', np.float32),
            ('NorthSouthAccuracyRMS', np.float32),
            ('MagnitudeRMS', np.float32),
            ('EastWestUncertaintyRMS', np.float32),
            ('NorthSouthUncertaintyRMS', np.float32),
            ('MagnitudeUncertaintyRMS', np.float32),
            ('EastWestMaxDeviation', np.float32),
            ('NorthSouthMaxDeviation', np.float32),
            ('MagnitudeMaxDeviation', np.float32),
            ('EastWestUncertaintyMax', np.float32),
            ('NorthSouthUncertaintyMax', np.float32),
            ('MagnitudeUncertaintyMax', np.float32)
        ]

        relative_accuracy = absolute_accuracy
        pixels_500_relative_accuracy = absolute_accuracy
        pixels_16_relative_accuracy = absolute_accuracy

        misregistration_residuals = [
            ('QualityInfoValidity', np.uint8),
            ('EastWestResidual', np.float32),
            ('NorthSouthResidual', np.float32),
            ('EastWestUncertainty', np.float32),
            ('NorthSouthUncertainty', np.float32),
            ('EastWestRMS', np.float32),
            ('NorthSouthRMS', np.float32),
            ('EastWestMagnitude', np.float32),
            ('NorthSouthMagnitude', np.float32),
            ('EastWestMagnitudeUncertainty', np.float32),
            ('NorthSouthMagnitudeUncertainty', np.float32)
        ]

        geometric_quality_status = [
            ('QualityNominal', np.uint8),
            ('NominalAbsolute', np.uint8),
            ('NominalRelativeToPreviousImage', np.uint8),
            ('NominalForREL500', np.uint8),
            ('NominalForREL16', np.uint8),
            ('NominalForResMisreg', np.uint8)
        ]

        record = [
            ('AbsoluteAccuracy', (absolute_accuracy, 12)),
            ('RelativeAccuracy', (relative_accuracy, 12)),
            ('500PixelsRelativeAccuracy', (pixels_500_relative_accuracy, 12)),
            ('16PixelsRelativeAccuracy', (pixels_16_relative_accuracy, 12)),
            ('MisregistrationResiduals', (misregistration_residuals, 12)),
            ('GeometricQualityStatus', (geometric_quality_status, 12))
        ]

        return record

    @property
    def timeliness_and_completeness(self):

        timeliness = [
            ('MaxDelay', np.float32),
            ('MinDelay', np.float32),
            ('MeanDelay', np.float32)
        ]

        completeness = [
            ('PlannedL15ImageLines', np.uint16),
            ('GeneratedL15ImageLines', np.uint16),
            ('ValidL15ImageLines', np.uint16),
            ('DummyL15ImageLines', np.uint16),
            ('CorruptedL15ImageLines', np.uint16)
        ]

        record = [
            ('Timeliness', timeliness),
            ('Completeness', (completeness, 12))
        ]

        return record


class HritPrologue(L15DataHeaderRecord):

    def get(self):

        # X bytes
        record = [
            ('SatelliteStatus', self.satellite_status),
            ('ImageAcquisition', self.image_acquisition),
            ('CelestialEvents', self.celestial_events),
            ('ImageDescription', self.image_description),
            ('RadiometricProcessing', self.radiometric_processing),
            ('GeometricProcessing', self.geometric_processing)
        ]

        return np.dtype(record).newbyteorder('>')


hrit_epilogue = np.dtype(
    Msg15NativeTrailerRecord().seviri_l15_trailer).newbyteorder('>')
hrit_prologue = HritPrologue().get()
impf_configuration = np.dtype(
    L15DataHeaderRecord().impf_configuration).newbyteorder('>')
native_header = Msg15NativeHeaderRecord().get()
native_trailer = Msg15NativeTrailerRecord().get()
